import matplotlib.pyplot as plt

# Data for the chart
epochs = [1, 2, 3]
accuracy = [0.75, 0.5, 0.5]
f1_score = [0.746, 0.381, 0.381]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(epochs, accuracy, label='Accuracy', marker='o')
plt.plot(epochs, f1_score, label='F1 Score', marker='s')

# Add labels, title, and legend
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.title('Model Performance Over Epochs')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
plt.savefig('model_performance.png')