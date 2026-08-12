[server]
# The default file watcher walks every imported module's __path__ to support
# hot-reload in dev mode. transformers imports 100+ optional vision
# submodules that try (and fail) to import torchvision, which floods the
# Streamlit Cloud logs with harmless-but-alarming tracebacks on every
# startup and file change. Disabling the watcher stops that noise; it has
# no effect on the app's actual behavior since Streamlit Cloud restarts the
# app on every git push anyway.
fileWatcherType = "none"

[browser]
gatherUsageStats = false
