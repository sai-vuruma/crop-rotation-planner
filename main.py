import numpy as np
import pandas as pd
import random
import os

import torch
from sklearn.model_selection import train_test_split
from collections import deque

import matplotlib.pyplot as plt
import seaborn as sns

from field import Field
from dqn import DQN

input_dim, output_dim = 6, 10
model_path = 'models/trained_dqn.pt'

def train_dqn(new_field, train_configs):
    #Initializing model parameters
    epochs = 1000
    learning_rate = 0.001
    gamma = 0.9
    epsilon = 0.1
    max_steps = 25 #don't exceed 25 the dataset isn't large enough
    episodes = 100
    batch_size = 4

    #Initialing DQN
    nnet = DQN(input_dim, output_dim)
    replay_buffer = deque(maxlen=10000)
    optimizer = torch.optim.Adam(nnet.parameters(), lr=learning_rate)
    loss_fn = torch.nn.HuberLoss()

    all_rewards = []
    all_losses = []

    #Training DQN
    nnet.train()
    for epoch in range(epochs):
        total_reward = 0
        train_losses = []

        for _ in range(episodes):
            region, soil_type = random.choice(train_configs)

            try:
                state = new_field.reset(region, soil_type)
            except:
                continue

            steps = 0
            while steps < max_steps:
                # Epsilon-greedy action selection
                if np.random.rand() < epsilon:
                    action = np.random.choice(output_dim)
                else:
                    with torch.no_grad():
                        q_values = nnet(torch.tensor(state, dtype=torch.float32))
                        action = int(torch.argmax(q_values).item())

                next_state, reward, done, info = new_field.step(action)
                replay_buffer.append((state, action, reward, next_state, done))

                state = next_state
                total_reward += reward
                steps += 1

                if len(replay_buffer) >= batch_size:
                    batch = random.sample(replay_buffer, batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    states = torch.tensor(states, dtype=torch.float32)
                    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
                    next_states = torch.tensor(next_states, dtype=torch.float32)
                    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

                    q_values = nnet(states).gather(1, actions)
                    with torch.no_grad():
                        next_q_values = nnet(next_states).max(1, keepdim=True)[0]
                        targets = rewards + (1 - dones) * gamma * next_q_values

                    loss = loss_fn(q_values, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_losses.append(loss.item())

        avg_loss = np.mean(train_losses) if train_losses else 0
        all_rewards.append(total_reward)
        all_losses.append(avg_loss)

        if (epoch+1) % (epochs/10) == 0:
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss} | Total Reward: {total_reward:.2f}")
        
    #Saving model checkpoint
    torch.save(nnet.state_dict(), model_path)
    
    #Plotting train performance
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(all_losses, label="Avg Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(all_rewards, label="Total Reward", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title("Total Reward per Epoch")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/train_performance.png')

    return nnet

myopic_lookup = {}

#Returns best action according to myopic policy
def myopic_policy(region, soil_type):
    key = (region, soil_type)
    if key in myopic_lookup:
        return myopic_lookup[key][0]
    else:
        #fail-safe
        return np.random.choice(output_dim)
    
#Evaluates learned policy against random, myopic  policies
def evaluate_policy(nnet, new_field, policy_fn, name="Policy", test_configs=None):
    episode_rewards = []
    episodes, max_steps = 100, 25

    for _ in range(episodes):
        region, soil_type = random.choice(test_configs)
        try:
            state = new_field.reset(region, soil_type)
        except:
            continue

        steps = 0
        total_reward = 0
        while steps < max_steps:
            if policy_fn == "random":
                action = np.random.choice(output_dim)
            elif policy_fn == "myopic":
                action = myopic_policy(region, soil_type)
            else:
                with torch.no_grad():
                    q_values = nnet(torch.tensor(state, dtype=torch.float32))
                    action = int(torch.argmax(q_values).item())

            next_state, reward, done, _ = new_field.step(action)
            state = next_state
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)

    return episode_rewards

def main():
    dataset = pd.read_csv("data/new_synthetic_agri_data_india_fixed.csv")
    new_field = Field(dataset)

    configs = new_field.environments
    train_configs, test_configs = train_test_split(configs, test_size=0.25, random_state=42)

    #Model training
    if not os.path.exists(model_path):
        print("-- Training DQN --")
        nnet = train_dqn(new_field, train_configs)
        print("-- Training complete --")
        print(f"Model saved at {model_path}")
    else:
        print(f"-- Using trained model at {model_path}")
        nnet = DQN(input_dim, output_dim)
        nnet.load_state_dict(torch.load(model_path, weights_only=True))

    #Evaluation
    nnet.eval()

    #Defining myopic policy
    grouped_rewards = new_field.world.df.groupby(["Region", "Soil Type", "action"])["reward"].mean().reset_index()

    for _, row in grouped_rewards.iterrows():
        key = (row["Region"], row["Soil Type"])
        if key not in myopic_lookup or row["reward"] > myopic_lookup[key][1]:
            myopic_lookup[key] = (int(row["action"]), row["reward"])
    
    dqn_rewards = evaluate_policy(nnet, new_field, "dqn", "DQN", test_configs=test_configs)
    random_rewards = evaluate_policy(nnet, new_field, "random", "Random", test_configs=test_configs)
    myopic_rewards = evaluate_policy(nnet, new_field, "myopic", "Myopic", test_configs=test_configs)

    plt.figure(figsize=(12,5))
    plt.plot(dqn_rewards, label="Our Policy")
    plt.plot(random_rewards, label="Random Policy")
    plt.plot(myopic_rewards, label="Myopic Policy")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode on Test set")
    plt.grid(True)
    plt.legend()
    plt.savefig('results/policy_comparison_line.png')

    data = {
        "Our Policy": dqn_rewards,
        "Random Policy": random_rewards,
        "Myopic Policy": myopic_rewards
    }

    plt.figure(figsize=(12,5))
    sns.boxplot(data=data)
    plt.title("Policy Comparison on Test Set")
    plt.ylabel("Total Reward per Episode")
    plt.savefig('results/policy_comparison_box.png')

    print("-- Evaluation complete on test set --")
    print("-- Results are stored in the results folder --")

if __name__ == "__main__":
    main()