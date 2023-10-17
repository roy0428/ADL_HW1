import matplotlib.pyplot as plt
import json
import ipdb

def main():
    completed_steps = []
    average_loss = []
    exact_match = []

    with open('result_2/training_curve.json', 'r') as json_file:
        curve_data = json.load(json_file)
        curve_data = curve_data["curve"]
        for data in curve_data:
            completed_steps.append(data["completed_steps"])
            average_loss.append(data["average_loss"])
            exact_match.append(data["exact_match_metric"])

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(completed_steps, average_loss)
    plt.xlabel("Completed Steps")
    plt.ylabel("Average Loss")
    plt.title("Loss Curve")

    plt.subplot(122)
    plt.plot(completed_steps, exact_match)
    plt.xlabel("Completed Steps")
    plt.ylabel("Exact Match (EM)")
    plt.title("EM Curve")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()