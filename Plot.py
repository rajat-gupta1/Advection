import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 

# Storing the three file names into a list and iterating over it
outputs = ["output", "output2", "output3"]

for output in outputs:

    df = pd.read_csv(output, header= None)
    df = df.iloc[:,:-1]

    np_df = np.array(df)

    fig, ax = plt.subplots()
    im = ax.imshow(np_df, interpolation = 'quadric')

    # Setting up title basis different cases
    if output == "output":
        ax.set_title("Initial State")
    elif output == "output2":
        ax.set_title("State at NT/2")
    else:
        ax.set_title("State at NT")

    # Storing graph in a file
    fig = ax.get_figure()
    fig.savefig(output + '.png')