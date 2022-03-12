# Git initiation at the start of each project
echo "# <project_name>" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote remove origin
git remote add origin https://github.com/vaibrainium/<project_name>.git
git push -u origin main

# Export all packages for transportation
conda env create --name recoveredenv --file environment.yml

# Import all packages
conda env update --prefix ./env --file environment.yml --prune

# Ignore folders
*.egg-info
<folder_name>