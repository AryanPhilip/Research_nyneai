use std::{fs, path::PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use nyne_er_core::{
    PairFeatureExtractor, ProfileRecord, SplitName, SourceType, TextProfileInput,
    assign_profile_splits, build_examples_for_profiles, load_benchmark_profiles,
    profile_from_text, profile_from_url, profiles_for_split, resolve_identities,
    resolve_live_profile, run_benchmark, search_and_build_profiles, search_person,
    train_hybrid_matcher,
};

#[derive(Debug, Parser)]
#[command(name = "nyne-er")]
#[command(about = "Rust CLI for nyne_er_lab")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Benchmark(BenchmarkArgs),
    Resolve(ResolveArgs),
    Search(SearchArgs),
}

#[derive(Debug, Args)]
struct BenchmarkArgs {
    #[arg(long, default_value = "real_curated_core")]
    dataset: String,
    #[arg(long, default_value = "7,11,17")]
    seeds: String,
    #[arg(long, default_value = "reports/benchmark_metrics_rust.json")]
    output: PathBuf,
}

#[derive(Debug, Args)]
struct ResolveArgs {
    #[arg(long)]
    profile_json: Option<PathBuf>,
    #[arg(long)]
    url: Option<String>,
    #[arg(long)]
    text_json: Option<PathBuf>,
    #[arg(long)]
    output: Option<String>,
    #[arg(long)]
    source_type: Option<String>,
    #[arg(long)]
    display_name_hint: Option<String>,
    #[arg(long, default_value = "real_curated_core")]
    dataset: String,
}

#[derive(Debug, Args)]
struct SearchArgs {
    #[arg(long)]
    name: String,
    #[arg(long, default_value_t = 6)]
    max_results: usize,
    #[arg(long)]
    output: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Benchmark(args) => run_benchmark_command(args),
        Commands::Resolve(args) => run_resolve_command(args),
        Commands::Search(args) => run_search_command(args),
    }
}

fn run_benchmark_command(args: BenchmarkArgs) -> Result<()> {
    let seeds = parse_seeds(&args.seeds)?;
    let report = run_benchmark(&args.dataset, "grouped_cv", Some(seeds))?;
    let payload = serde_json::to_string_pretty(&report)?;
    match args.output.to_str() {
        Some("stdout") | Some("-") => {
            println!("{payload}");
            Ok(())
        }
        _ => write_output(args.output, &payload),
    }
}

fn run_resolve_command(args: ResolveArgs) -> Result<()> {
    let profile = match (&args.profile_json, &args.url, &args.text_json) {
        (Some(path), None, None) => {
            let content = fs::read_to_string(path)?;
            let profile: ProfileRecord = serde_json::from_str(&content)?;
            profile.validated()?
        }
        (None, Some(url), None) => {
            let source_type = args
                .source_type
                .as_deref()
                .map(parse_source_type)
                .transpose()?
                .unwrap_or(SourceType::PersonalSite);
            profile_from_url(url, source_type.as_str(), args.display_name_hint.as_deref())
                .context("failed to fetch or parse URL into a profile")?
        }
        (None, None, Some(path)) => {
            let content = fs::read_to_string(path)?;
            let input: TextProfileInput = serde_json::from_str(&content)?;
            profile_from_text(input)?
        }
        _ => bail!("Provide exactly one of --profile-json, --url, or --text-json"),
    };

    let profiles = load_benchmark_profiles(&args.dataset)?;
    let split_map = assign_profile_splits(&profiles);
    let train_profiles = profiles_for_split(&profiles, &split_map, SplitName::Train);
    let val_profiles = profiles_for_split(&profiles, &split_map, SplitName::Val);
    let extractor = PairFeatureExtractor::default().fit(&train_profiles)?;
    let train_examples = build_examples_for_profiles(&train_profiles, &extractor, SplitName::Train)?;
    let val_examples = build_examples_for_profiles(&val_profiles, &extractor, SplitName::Val)?;
    let matcher = train_hybrid_matcher(&train_examples, &val_examples, &extractor, None)?;
    let all_examples = build_examples_for_profiles(&profiles, &extractor, SplitName::Test)?;
    let (identities, _) = resolve_identities(&profiles, &all_examples, &matcher, &extractor)?;
    let result = resolve_live_profile(&profile, &profiles, &extractor, &matcher, &identities)?;
    let payload = serde_json::to_string_pretty(&result)?;
    write_maybe_stdout(args.output.as_deref(), &payload)
}

fn run_search_command(args: SearchArgs) -> Result<()> {
    let payload = serde_json::to_string_pretty(&serde_json::json!({
        "hits": search_person(&args.name, args.max_results)?,
        "profiles": search_and_build_profiles(&args.name, args.max_results)?,
    }))?;
    write_maybe_stdout(args.output.as_deref(), &payload)
}

fn parse_seeds(raw: &str) -> Result<Vec<u64>> {
    raw.split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.trim().parse::<u64>().context("invalid seed"))
        .collect()
}

fn parse_source_type(raw: &str) -> Result<SourceType> {
    match raw {
        "github" => Ok(SourceType::Github),
        "personal_site" => Ok(SourceType::PersonalSite),
        "conference_bio" => Ok(SourceType::ConferenceBio),
        "company_profile" => Ok(SourceType::CompanyProfile),
        "podcast_guest" => Ok(SourceType::PodcastGuest),
        "huggingface" => Ok(SourceType::Huggingface),
        _ => bail!("unsupported source type '{raw}'"),
    }
}

fn write_maybe_stdout(output: Option<&str>, content: &str) -> Result<()> {
    match output {
        None | Some("stdout") | Some("-") => {
            println!("{content}");
            Ok(())
        }
        Some(path) => write_output(PathBuf::from(path), content),
    }
}

fn write_output(path: PathBuf, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, content)?;
    Ok(())
}
