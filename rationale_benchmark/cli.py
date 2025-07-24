"""Command-line interface for the rationale benchmark tool."""

import click


@click.command()
@click.option(
    "--questionnaire", help="Single questionnaire to run (filename without .yaml)"
)
@click.option("--questionnaires", help="Comma-separated list of questionnaires to run")
@click.option(
    "--llm-config",
    default="default-llms",
    help="LLM configuration to use (filename without .yaml, defaults to 'default-llms')",
)
@click.option("--models", help="Comma-separated list of specific models to test")
@click.option(
    "--output", type=click.Path(), help="Output file for results (JSON format)"
)
@click.option(
    "--list-questionnaires", is_flag=True, help="List all available questionnaires"
)
@click.option(
    "--list-llm-configs", is_flag=True, help="List all available LLM configurations"
)
@click.option(
    "--config-dir",
    type=click.Path(exists=True),
    default="./config",
    help="Custom path to configuration directory (default: ./config)",
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def main(
    questionnaire,
    questionnaires,
    llm_config,
    models,
    output,
    list_questionnaires,
    list_llm_configs,
    config_dir,
    verbose,
):
    """Rationale Benchmark for Large Language Models.

    A configurable benchmark tool designed to evaluate large language models (LLMs)
    based on their similarity to human rationale, including human intuitions and
    psychological reasoning patterns.
    """
    if verbose:
        click.echo("Verbose mode enabled")

    if list_questionnaires:
        click.echo("Available questionnaires:")
        click.echo("  (Implementation pending - questionnaire discovery)")
        return

    if list_llm_configs:
        click.echo("Available LLM configurations:")
        click.echo("  (Implementation pending - LLM config discovery)")
        return

    # Main benchmark execution logic will be implemented here
    click.echo("🚀 Rationale Benchmark Tool")
    click.echo("=" * 40)

    if questionnaire:
        click.echo(f"Running questionnaire: {questionnaire}")
    elif questionnaires:
        questionnaire_list = [q.strip() for q in questionnaires.split(",")]
        click.echo(f"Running questionnaires: {', '.join(questionnaire_list)}")
    else:
        click.echo("Running all available questionnaires")

    click.echo(f"Using LLM config: {llm_config}")

    if models:
        model_list = [m.strip() for m in models.split(",")]
        click.echo(f"Testing models: {', '.join(model_list)}")

    if output:
        click.echo(f"Output will be saved to: {output}")

    click.echo("\n⚠️  Implementation in progress...")
    click.echo(
        "This is a placeholder CLI. Core functionality will be implemented soon."
    )


if __name__ == "__main__":
    main()
