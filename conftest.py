# Presence of this file at the repository root puts the root on sys.path for
# pytest's default (prepend) import mode, so `quant_assistant` and `web` import
# cleanly whether tests are run as `pytest` or `python -m pytest`, in CI or
# locally, without an editable install.
