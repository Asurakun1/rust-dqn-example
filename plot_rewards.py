import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file our Rust program created
try:
    data = pd.read_csv('rewards.csv')
except FileNotFoundError:
    print("Error: rewards.csv not found. Run the Rust training program first.")
    exit()

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(data['episode'], data['reward'], alpha=0.5, label='Reward per Episode')

# --- A Helpful Trick: Moving Average ---
# Raw reward data is often very "noisy". A moving average helps see the real trend.
moving_avg = data['reward'].rolling(window=50).mean()
plt.plot(data['episode'], moving_avg, color='red', linewidth=2, label='50-Episode Moving Avg')
# -----------------------------------------

plt.title('DQN Agent Learning Progress')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
plt.show() # Display the plot