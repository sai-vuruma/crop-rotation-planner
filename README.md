<h1>RL-driven Crop Rotation Planner</h1>
<p>Crop rotation is a technique that farmers use to manage soil nutrients - crops are alternated each season to avoid overuse. We propose a reinforcement learning framework to predict crop rotation sequences, balancing short-term yield and long-term soil health through Deep Q-Networks (DQN). Our approach performs better than a random policy and worse than a myopic policy in optimizing long-term reward.</p>

Dataset is taken from Kaggle.
<cite>https://www.kaggle.com/datasets/abhaysasthas/crop-rotation-dataset</cite>

The repo is organized as,
<ul>
<li>Folders:
    <li>data/ contains the dataset</li>
    <li>models/ contains the trained DQN model. Delete the trained_dqn.pt file to retrain.</li>
    <li>results/ contains visualizations of the model's performance during training and during evaluation. We compare the learned policy against both random and myopic policies.</li>
</li>
<li>Files:
    <li>dqn.py contains the DQN definition. The network architecture can be modified here.</li>
    <li>env.yaml contains the requirements for the conda env needed to run the code.</li>
    <li>field.py and world.py contain code to encode the world (region-soil type) and field characteristics. These files layout the POMDP.</li>
    <li>main.py is the main wrapper that brings it all together. Executing this one file will run all of the code.</li>
</li>
</ul>

Follow these instructions to replicate the results:

<ol>
<li>Create the conda enviroment</li>
<code>conda env create -f env.yaml</code>

<li>Activate the conda environment</li>
<code>conda activate crop-rotation</code>

<li>Execute the main.py file</li>
<code>python main.py</code>

<li>Check out the results in the results/ folder</li>
</ol>

The results are generated using the saved model in the models/ folder. To retrain the model, delete the model in the models/ folder and repeat step 3 from the above list.