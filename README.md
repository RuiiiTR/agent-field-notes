# Agent Field Notes

A Zensical-powered learning blog about building and evaluating AI agents.

## Inspiration

The documentation-style shell and long-form technical writing approach were inspired by [One Poem Suffices](https://keli-wen.github.io/One-Poem-Suffices/) by [Keli Wen](https://github.com/keli-wen). Agent Field Notes uses its own identity, content, experiments, and article structure.

## Local preview

```bash
python -m venv .venv
source .venv/bin/activate
pip install zensical
zensical serve
```

## Build and publish

Run `zensical build --clean`, or push to `main` to deploy through GitHub Pages.

Live site: <https://ruiiitr.github.io/agent-field-notes/>

## Writing a post

Reusable writing templates are stored outside the published `docs/` directory:

- `templates/blog-post.md` — substantial technical article
- `templates/deep-dive-blog-post.md` — long-form “one article is enough” tutorial with mechanism analysis, practice, debugging, and references
- `templates/learning-log.md` — short weekly learning update

Copy a template into the appropriate section and rename it:

```text
docs/agents/          ReAct, planning, reflection, tool calling
docs/context/         Memory, RAG, context engineering
docs/engineering/     Frameworks, evaluation, safety, capstone
docs/learning-log/    Weekly progress and short reflections
```

After creating an article, add its path to `zensical.toml` so it appears in the sidebar.
