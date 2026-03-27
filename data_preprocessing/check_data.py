import os

def check_datasets():
    datasets = ['bdd100k', 'bdd100k_tiny', 'bdd100k_tiny_data']
    print("Checking dataset directories in data_preprocessing/...")
    for ds in datasets:
        path = os.path.join('data_preprocessing', ds)
        if os.path.exists(path):
            print(f"[OK] Found {ds}")
        else:
            print(f"[MISSING] {ds} not found. (Expected as it's ignored by Git)")

if __name__ == "__main__":
    check_datasets()