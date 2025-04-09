c = get_config()

# Use local directory for runtime files
c.ServerApp.runtime_dir = '.jupyter/runtime'

# Don't try to open browser automatically
c.ServerApp.open_browser = False

# Allow all IP addresses to connect
c.ServerApp.ip = '0.0.0.0'

# Set the data directory to a local path
c.ServerApp.data_dir = '.jupyter/data'

# Set the notebook directory to the current directory
c.ServerApp.notebook_dir = '.' 