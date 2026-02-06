import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_distribution(data_path, title):
    categories = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    counts = {cat: len(os.listdir(os.path.join(data_path, cat))) for cat in categories}
    
    df = pd.DataFrame(list(counts.items()), columns=['Disease', 'Image_Count'])
    df = df.sort_values(by='Image_Count', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Disease', y='Image_Count', palette='magma')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the report for the presentation
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    print(f"Graph saved as {title.replace(' ', '_').lower()}.png")
    plt.show()

if __name__ == "__main__":
    train_path = "data/train_data"
    test_path = "data/test_data"
    
    if os.path.exists(train_path):
        plot_distribution(train_path, "Training Data Distribution")
    if os.path.exists(test_path):
        plot_distribution(test_path, "Testing Data Distribution")