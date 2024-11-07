from flask import Flask, render_template, request, redirect, url_for
import random
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from scipy.stats import kurtosis, skew
from scipy.stats import ks_2samp

app = Flask(__name__)

# Generate 500 random numbers from a normal distribution centered around 10 with a standard deviation of 4
original_numbers = np.clip(np.random.normal(10, 4, 500), 0, 20).round().astype(int).tolist()

def create_plot(data, title, display_kurtosis, display_skewness, overlay_curve=True):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=range(22), edgecolor='black', density=True, alpha=0.6, color='steelblue')
    
    # Overlay normal distribution curve based on data if requested
    if overlay_curve:
        mean, std_dev = np.mean(data), np.std(data)
        x = np.linspace(0, 20, 100)
        y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
        plt.plot(x, y, color='red', linewidth=2)
    
    plt.title(title)
    plt.xlabel("Number")
    plt.ylabel("Frequency")
    plt.figtext(0.15, -0.05, f"Kurtosis: {display_kurtosis:.2f}, Skewness: {display_skewness:.2f}", ha="left")
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches="tight")
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

def create_comparison_plot(original_data, updated_data, title, display_kurtosis, display_skewness, include_legend=False):
    plt.figure(figsize=(7.5, 4.5))  # Scaled down size
    
    # Plot the updated histogram
    plt.hist(updated_data, bins=range(22), edgecolor='black', density=True, alpha=0.6, color='steelblue')
    
    # Calculate the original and updated means and standard deviations
    original_mean, original_std_dev = np.mean(original_data), np.std(original_data)
    updated_mean, updated_std_dev = np.mean(updated_data), np.std(updated_data)
    
    # Generate x values for both curves
    x = np.linspace(0, 20, 100)
    
    # Original bell curve (dotted line)
    y_original = (1 / (original_std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - original_mean) / original_std_dev) ** 2)
    plt.plot(x, y_original, 'r--', linewidth=2, label="Original Curve" if include_legend else None)
    
    # Updated bell curve (solid line)
    y_updated = (1 / (updated_std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - updated_mean) / updated_std_dev) ** 2)
    plt.plot(x, y_updated, 'r-', linewidth=2, label="Updated Curve" if include_legend else None)
    
    # Add title, labels, and conditional legend
    plt.title(title)
    plt.xlabel("Number")
    plt.ylabel("Frequency")
    if include_legend:
        plt.legend(loc="upper right")
    plt.figtext(0.15, -0.08, f"Kurtosis: {display_kurtosis:.2f}, Skewness: {display_skewness:.2f}", ha="left")
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches="tight")
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        global original_numbers
        user_input = request.form.get("numbers")
        try:
            new_numbers = [int(num) for num in user_input.split() if num.isdigit() and 0 <= int(num) <= 20]
            if not all(num.isdigit() and 0 <= int(num) <= 20 for num in user_input.split()):
                raise ValueError
        except ValueError:
            return render_template("index.html", show_plots=False, error="Input valid values. This app accepts only the whole numbers between 0 and 20.")
        
        updated_numbers = original_numbers + new_numbers
        
        # Calculate statistics
        original_kurtosis = round(kurtosis(original_numbers), 3)
        original_skewness = round(skew(original_numbers), 3)
        updated_kurtosis = round(kurtosis(updated_numbers), 3)
        updated_skewness = round(skew(updated_numbers), 3)

        original_mean = round(np.mean(original_numbers), 3)
        updated_mean = round(np.mean(updated_numbers), 3)
        original_std_dev = round(np.std(original_numbers), 3)
        updated_std_dev = round(np.std(updated_numbers), 3)
        
        # Perform KS Test
        ks_statistic, ks_p_value = ks_2samp(original_numbers, updated_numbers)
        ks_statistic = round(ks_statistic, 3)
        ks_p_value = round(ks_p_value, 3)

        # Create plots with updated size and spacing
        original_plot_url = create_comparison_plot(original_numbers, original_numbers, "Original Distribution", original_kurtosis, original_skewness, include_legend=False)
        updated_plot_url = create_comparison_plot(original_numbers, updated_numbers, "Updated Distribution with Overlayed Curves", updated_kurtosis, updated_skewness, include_legend=True)

        # Prepare KS test results text
        ks_test_results = f"KS Test between original and updated distributions: KS Statistic = {ks_statistic}, p-value = {ks_p_value}"
        
        return render_template("index.html", 
                               original_plot_url=original_plot_url, 
                               updated_plot_url=updated_plot_url, 
                               original_kurtosis=original_kurtosis, 
                               original_skewness=original_skewness,
                               updated_kurtosis=updated_kurtosis, 
                               updated_skewness=updated_skewness, 
                               original_mean=original_mean,
                               updated_mean=updated_mean,
                               original_std_dev=original_std_dev,
                               updated_std_dev=updated_std_dev,
                               ks_test_results=ks_test_results,
                               show_plots=True, 
                               new_numbers=new_numbers)
    
    return render_template("index.html", show_plots=False)


@app.route("/reset")
def reset():
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
