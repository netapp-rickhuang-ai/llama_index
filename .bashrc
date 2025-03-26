
source /Users/yinray/.docker/init-bash.sh || true # Added by Docker Desktop 
# Clean up the PATH variable 
# The next line updates PATH for the Google Cloud SDK. 
if [ -f 
'/Users/yinray/Library/CloudStorage/OneDrive-NetAppInc/workspace/google-cloud-sdk/path.bash.inc' 
]; then . 
'/Users/yinray/Library/CloudStorage/OneDrive-NetAppInc/workspace/google-cloud-sdk/path.bash.inc'; 
fi export 
PATH="/Users/yinray/.local/share/virtualenvs/rag-sample-mistral-tJXqdaBL/bin:/Users/yinray/.docker/bin:/Users/yinray/Library/CloudStorage/OneDrive-NetAppInc/workspace/google-cloud-sdk/bin:/opt/anaconda3/bin:/opt/anaconda3/condabin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/Apple/usr/bin" 

# The next line enables shell command completion for gcloud. 

if [ -f 
'/Users/yinray/Library/CloudStorage/OneDrive-NetAppInc/workspace/google-cloud-sdk/completion.bash.inc' 
]; then . 
'/Users/yinray/Library/CloudStorage/OneDrive-NetAppInc/workspace/google-cloud-sdk/completion.bash.inc'; 
fi 
# Save and exit the editor (Ctrl+O, Enter, Ctrl+X for nano) # Reload the shell configuration
source ~/.bashrc  # or source ~/.bash_profile

export PATH="/Users/yinray/.local/share/virtualenvs/rag-sample-mistral-tJXqdaBL/bin:/Users/yinray/.docker/bin:/Users/yinray/Library/CloudStorage/OneDrive-NetAppInc/workspace/google-cloud-sdk/bin:/opt/anaconda3/bin:/opt/anaconda3/condabin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/Apple/usr/bin"

# [NOTE_TO_SELF] I've tried the following:

pip install 'tiktoken' 
pipenv graph
pipenv shell
history

# [TODO] Try ing diff dir/system level
