import kagglehub

# Download latest version
path = kagglehub.dataset_download("tr1gg3rtrash/cars-2022-dataset")

print("Path to dataset files:", path)