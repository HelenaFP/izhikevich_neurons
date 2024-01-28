# numpy
import numpy as np
import numpy.random as rn

# plotting libraries
import plotly.express as px
import plotly.graph_objects as go

# utilities
from tqdm import tqdm
import os

# Izhikevich neuron class
from iznetwork import IzNetwork

class SWMNetwork:
    """Small-world modular network of Izhikevich neurons.

    This class represents a small-world modular network of Izhikevich neurons with configurable parameters.
    It allows you to create, simulate, and analyze spiking neural network models.

    Attributes:
    - nb_modules (int): Number of modules in the network.
    - e_neurons (int): Number of excitatory neurons in each module.
    - i_neurons (int): Number of inhibitory neurons in the network.
    - nb_connections (int): Number of connections per module.
    - p (float): The rewiring probability for excitatory to excitatory connections.
    - time_simulation (int): Total duration of the simulation in time steps.
    - net (IzNetwork): Configured spiking neural network.
    - v (ndarray): Membrane potential activity during the simulation.

    Methods:
    - create_weight_matrix(): Create the weight matrix for the network.
    - create_delay_matrix(): Create the delay matrix for the network.
    - create_network(): Create and configure the spiking neural network.
    - plot_connectivity(): Generate and save a connectivity matrix plot.
    - run_simulation(): Run a simulation of the network.
    - compute_mean_firing_rates(): Calculate mean firing rates for each module during a simulation.
    - plot_raster(): Generate and save a raster plot of excitatory neuron spiking activity.
    - plot_firing_rate(): Generate and save a plot of mean firing rates in network modules.

    """

    def __init__(self, nb_modules, e_neurons, i_neurons, nb_connections, p, time_simulation):
        """ Initialize a Small-world Modular Network of Izhikevich Neurons.

        Attributes:
        - nb_modules (int): Number of modules in the network.
        - e_neurons (int): Number of excitatory neurons in each module.
        - i_neurons (int): Number of inhibitory neurons in the network.
        - 'total_neurons': Total number of neurons in the network.
        - nb_connections (int): Number of connections per module.
        - p (float): The rewiring probability for excitatory to excitatory connections.
        - time_simulation (int): Total duration of the simulation in time steps.
        - net (IzNetwork): Configured spiking neural network.
        - v (ndarray): Membrane potential activity during the simulation.

        Returns:
        - SWMNetwork: An instance of the Small-world Modular Network of Izhikevich Neurons.

        Example:
        >>> network = SWMNetwork(nb_modules=8, e_neurons=100, i_neurons=200, nb_connections=100, p=0.1, time_simulation=1000)

        """

        self.nb_modules = nb_modules
        self.e_neurons  = e_neurons
        self.i_neurons  = i_neurons
        self.total_neurons = nb_modules*e_neurons + i_neurons
        self.nb_connections = nb_connections
        self.p  = p
        self.time_simulation  = time_simulation
        self.net = self.create_network()
        self.v = np.zeros((time_simulation, e_neurons*nb_modules + i_neurons))

    def create_weight_matrix(self):
        """Create the weight matrix for the small-world modular network of Izhikevich neurons.

        This function initializes the weight matrix with specific connection patterns and strengths.
        The matrix is created for both excitatory and inhibitory neurons, considering intra-community and inter-community connections.
        The weight matrix is stored in the 'W' attribute of the object for later use.

        Usage:
        Call this function to generate the weight matrix for the network. The resulting weight matrix is stored in the object's 'W' attribute.

        Notes:
        - 'e_e_factor': Strength of excitatory to excitatory connections within a module.
        - 'e_i_factor': Strength of excitatory to inhibitory connections.
        - 'i_e_factor': Strength of inhibitory to excitatory connections.
        - 'i_i_factor': Strength of inhibitory to inhibitory connections.

        """
        
        nb_modules = self.nb_modules
        nb_connections = self.nb_connections
        p = self.p
        e_e_factor = 17
        e_i_factor = 50
        i_e_factor = 2
        i_i_factor = 1

        oneBlock  = np.ones((100, 100))
        zeroBlock = np.zeros((100, 100))

        # Block [i,j] is the connection from population i to population j
        # Last 2 blocks contain inhibitory neurons
        W = np.bmat([[zeroBlock, zeroBlock,  zeroBlock, zeroBlock, zeroBlock, zeroBlock,  zeroBlock, zeroBlock, zeroBlock, zeroBlock],
                     [zeroBlock, zeroBlock,  zeroBlock,  zeroBlock, zeroBlock, zeroBlock,  zeroBlock, zeroBlock, zeroBlock, zeroBlock],
                     [zeroBlock, zeroBlock,  zeroBlock,  zeroBlock, zeroBlock, zeroBlock,  zeroBlock, zeroBlock, zeroBlock, zeroBlock],
                     [zeroBlock, zeroBlock,  zeroBlock,  zeroBlock, zeroBlock, zeroBlock,  zeroBlock, zeroBlock, zeroBlock, zeroBlock],
                     [zeroBlock, zeroBlock,  zeroBlock,  zeroBlock, zeroBlock, zeroBlock,  zeroBlock, zeroBlock, zeroBlock, zeroBlock],
                     [zeroBlock, zeroBlock,  zeroBlock,  zeroBlock, zeroBlock, zeroBlock,  zeroBlock, zeroBlock, zeroBlock, zeroBlock],
                     [zeroBlock, zeroBlock,  zeroBlock,  zeroBlock, zeroBlock, zeroBlock,  zeroBlock, zeroBlock, zeroBlock, zeroBlock],
                     [zeroBlock, zeroBlock,  zeroBlock,  zeroBlock, zeroBlock, zeroBlock,  zeroBlock, zeroBlock, zeroBlock, zeroBlock],
                     [i_e_factor*oneBlock, i_e_factor*oneBlock,  i_e_factor*oneBlock,  i_e_factor*oneBlock, i_e_factor*oneBlock, i_e_factor*oneBlock, i_e_factor*oneBlock, i_e_factor*oneBlock, i_i_factor*oneBlock, i_i_factor*oneBlock],
                     [i_e_factor*oneBlock, i_e_factor*oneBlock,  i_e_factor*oneBlock,  i_e_factor*oneBlock, i_e_factor*oneBlock, i_e_factor*oneBlock, i_e_factor*oneBlock, i_e_factor*oneBlock, i_i_factor*oneBlock, i_i_factor*oneBlock],
                    ])
        
        
        # multiply last 2 blocks of W by random numbers between -1 and 0
        W = np.array(W)
        W[800:, :] = W[800:, :] * rn.uniform(-1, 0, (200, 1000))

        # remove self connections for inhibitory neurons by setting diagonal values to 0
        np.fill_diagonal(W[800:, 800:], 0)


        # excitatory to excitatory connections
        # Each module has 1000 randomly assigned one-way excitatory-to-excitatory connections
        # print("excitatory to excitatory connections")
        for i in range(8):
            # Generate s and t arrays
            s = i*100 + np.random.randint(0, 100, 2000)  # Oversample to ensure we get enough unique pairs
            t = i*100 + np.random.randint(0, 100, 2000)
            # Pair up s and t
            pairs = np.vstack([s, t]).T

            # Make pairs unique
            unique_pairs = np.unique(pairs, axis=0)

            # Remove self-connections
            unique_pairs = unique_pairs[unique_pairs[:, 0] != unique_pairs[:, 1]]

            # If we didn't get enough unique pairs, keep adding pairs
            while unique_pairs.shape[0] < 1000:
                s = i*100 + np.random.randint(0, 100, 2000)
                t = i*100 + np.random.randint(0, 100, 2000)
                pairs_next = np.vstack([s, t]).T
                unique_pairs_next = pairs_next[pairs_next[:, 0] != pairs_next[:, 1]]
                unique_pairs = np.unique(np.concatenate((unique_pairs, unique_pairs_next)), axis=0)

            # Trim to 1000 pairs
            rn.shuffle(unique_pairs)
            unique_pairs = unique_pairs[:1000]
            W[unique_pairs[:, 0], unique_pairs[:, 1]] = e_e_factor*1


        # print("rewiring excitatory to excitatory connections")
        # rewire excitatory to excitatory connections
        # Each existing (intracommunity) edge is considered, and with probability p is rewired as an edge between communities
        src, tgt = np.where(W[:800, :800] > 0)

        # get module source belong to
        source_module = src // 100

        # Generate random numbers for all source-target pairs
        random_numbers = np.random.random(len(src))

        # Find the indices where the random number is less than p
        rewire_indices = np.where(random_numbers < p)

        # Set the corresponding elements in W to 0
        W[src[rewire_indices], tgt[rewire_indices]] = 0

        target_modules = [0,1,2,3,4,5,6,7]
        # Pick new module to rewire to at random
        for i in range(8):
            # select all indices where the source module is i
            indices = np.where(source_module[rewire_indices] == i)
            # Pick new module to rewire to at random
            # Oversample to ensure we get enough unique pairs
            targets = np.random.choice(target_modules, 2*len(indices[0]))*100 + rn.randint(0, 100, 2*len(indices[0]))
            # get unique values
            unique_targets = np.unique(targets)
            # If we didn't get enough unique pairs, keep adding pairs
            while unique_targets.shape[0] < len(indices[0]):
                targets = np.random.choice(target_modules, 2*len(indices[0]))*100 + rn.randint(0, 100, 2*len(indices[0]))
                targets_next = np.unique(targets)
                unique_targets = np.unique(np.concatenate((unique_targets, targets_next)))
            # Trim to len(indices[0]) pairs
            rn.shuffle(unique_targets)
            unique_targets = unique_targets[:len(indices[0])]
            # rewire
            W[src[rewire_indices[0][indices]], unique_targets] = e_e_factor*1


        # print("excitatory to inhibitory connections")
        # Generate all targets, sources, and source modules at once
        targets = np.repeat(np.arange(8*100, 8*100 + 200), 4)
        source_modules = np.repeat(np.random.randint(0, 8, size=targets.shape[0]//4), 4)

        for i in range(8):
            # select all indices where the source module is i
            indices = np.where(source_modules == i)
            # select source values
            sources = i*100 + np.random.randint(0, 100, size=len(indices[0]))
            # rewire
            # multiply by random number between 0 and 1
            W[sources, targets[indices]] = e_i_factor*1*rn.uniform(0, 1, size=len(indices[0]))

        self.W = W


    def create_delay_matrix(self):
        """Create the delay matrix for the small-world modular network of Izhikevich neurons.

        This function generates the delay matrix, specifying conduction delays between neurons based on their connection types.
        The delays are set to specific values for different connection types, and excitatory-to-excitatory connections have random delays between 1 and 20.
        The delay matrix is stored in the 'D' attribute of the object for later use.

        Usage:
        Call this function to create the delay matrix for the network. The resulting delay matrix is stored in the object's 'D' attribute.

        Notes:
        - 'W': The weight matrix of the network (assumed to be already created).
        - 'Dmax': Maximum conduction delay in the network.
        - 'D': The delay matrix to be created.

        """

        # delays
        # excitatory to excitatory: random 1 to 20
        # excitatory to inhibitory: 1
        # inhibitory to excitatory: 1
        # inhibitory to inhibitory: 1

        # create a matrix of delays with values of 1 
        D = np.ones(self.W.shape, dtype=int)

        # Create a mask for the excitatory to excitatory connections
        mask = np.zeros(D.shape, dtype=bool)
        mask[:8*100, :8*100] = True

        # Find the indices where the mask is True
        indices = np.where(mask)

        # Assign random integers to the excitatory to excitatory connections
        D[indices] = np.random.randint(1, 20, size=len(indices[0]))

        self.D = D


    def create_network(self):
        """Create a small-world modular network of Izhikevich neurons and configure its parameters.

        This function creates a spiking neural network using the Izhikevich neuron model and configures its parameters, 
        including weights, delays, and neuron-specific parameters such as membrane potential and recovery variable.
        The function returns a configured IzNetwork object that can be used for simulations and further analysis.

        Returns:
        - net (IzNetwork): The configured spiking neural network.

        Usage:
        Call this function to create and configure a spiking neural network. The network is configured with specific weight and delay matrices, as well as neuron-specific parameters.

        Notes:
        - 'Dmax': Maximum conduction delay in the network.
        - 'W': Weight matrix for the network (assumed to be already created).
        - 'D': Delay matrix for the network (assumed to be already created).
        - 'a_e', 'b_e', 'c_e', 'd_e': Parameters for excitatory neurons.
        - 'a_i', 'b_i', 'c_i', 'd_i': Parameters for inhibitory neurons.

        """

        Dmax = 20                # Maximum conduction delay
        net = IzNetwork(self.total_neurons, Dmax)

        self.create_weight_matrix()
        net.setWeights(self.W)

        self.create_delay_matrix()
        net.setDelays(self.D)
        
        a_e = 0.02
        b_e = 0.2
        c_e = -65
        d_e = 8
        a_i = 0.02
        b_i = 0.25
        c_i = -65
        d_i = 2

        a = np.concatenate((a_e*np.ones(8*100), a_i*np.ones(200)))
        b = np.concatenate((b_e*np.ones(8*100), b_i*np.ones(200)))
        c = np.concatenate((c_e*np.ones(8*100), c_i*np.ones(200)))
        d = np.concatenate((d_e*np.ones(8*100), d_i*np.ones(200)))
  
        net.setParameters(a, b, c, d)

        return net

    
    def plot_connectivity(self):
        """Create a connectivity matrix plot and save it as an image and HTML.

        This function generates a connectivity matrix plot from the weight matrix 
        and saves it as both an image (SVG) and an interactive HTML plot. 
        The plot visualizes the connectivity pattern between neurons.
        The generated plots are saved in the 'plots' and 'plots_html' directories, 
        and the file names include the value of 'p' in their names.

        Notes:
        - 'W': Weight matrix of the network (assumed to be already created).
        - 'p': The rewiring probability for excitatory to excitatory connections.

        """

        connections = np.where(self.W != 0, 1, self.W)
        
        fig = px.imshow(connections, color_continuous_scale='gray_r')
        if not os.path.exists("plots"):
            os.mkdir("plots")
        if not os.path.exists("plots_html"):
            os.mkdir("plots_html")
        
        fig.write_image(f"plots/matrix_connectivity_p{self.p}.svg")
        fig.write_html(f"plots_html/matrix_connectivity_p{self.p}.html")
        
    
    def run_simulation(self):
        """Run a simulation of the network.

        This function conducts a simulation of the configured spiking neural network over a specified duration. 
        It generates the network's activity, including membrane potentials, and stores the results in the 'v' attribute.

        Usage:
        Call this function to run a simulation of the configured spiking neural network. 
        The simulation results, including membrane potentials, are stored in the 'v' attribute.

        Notes:
        - 'time_simulation': The total duration of the simulation in time steps.
        - 'e_neurons': Number of excitatory neurons per module.
        - 'i_neurons': Number of inhibitory neurons.
        - 'nb_modules': Number of modules in the network.
        - 'nb_connections': Number of connections per module.
        - 'net': The configured spiking neural network (assumed to be already created).
        
        """

        V = np.zeros((self.time_simulation, self.total_neurons))
        for t in tqdm(range(self.time_simulation)):
            # for excitatory neurons, induce spikes following a poisson distribution
            poisson = rn.poisson(0.01, self.e_neurons*self.nb_modules)
            # When this process produces a value > 0, set  current to 15
            poisson[poisson > 0] = 15
            # concatenate this with the number of inhibitory neurons
            poisson = np.concatenate((poisson, np.zeros(self.i_neurons)))
            self.net.setCurrent(poisson)
            self.net.update()
            V[t,:], _ = self.net.getState()

        self.v = V

        return
    
    def compute_mean_firing_rates(self):
        """ Calculate the mean firing rates in each module of the network during a simulation.

        This function computes the mean firing rates in each of the eight modules of the spiking neural network during a 1000ms simulation run. 
        The firing rates are calculated by downsampling the activity to obtain the mean by computing the average number of firings in 50ms windows shifted every 20ms. 
        This results in 50 data points for each module.
        The computed mean firing rates are stored in the 'firing_rates' attribute for further analysis or visualization.

        Usage:
        Call this function to compute the mean firing rates for each module during the simulation. The results are stored in the 'firing_rates' attribute.

        Notes:
        - 'v': The membrane potential activity of the network during the simulation (assumed to be already available).
        - 'window_size': The size of the analysis window in milliseconds.
        - 'step_size': The time interval between consecutive analysis windows.
        - 'firing_rates': List of mean firing rates for each module.

        """

        V = np.where(self.v > 29, 1, 0)
        window_size = 50
        step_size = 20

        firing_rates = []

        # loop through modules
        for i in range(8):
            # select V of that module
            v_module = V[:, i*100:i*100+100]

            num_windows = (v_module.shape[0] - window_size) // step_size + 1
            mean_firing_rates = np.zeros((num_windows))

            for j in range(num_windows):
                start = j * step_size
                end = start + window_size
                window = v_module[start:end]
                # take mean per neuron
                mean_firing_per_neuron = np.mean(window, axis=0)
                # Take the mean across all neurons in the window
                mean_firing_rates[j]= mean_firing_per_neuron.mean() * 1000

            firing_rates.append(mean_firing_rates)

        self.firing_rates = firing_rates

        return
    

    def plot_raster(self):
        """ Create a raster plot of excitatory neuron spiking activity during a simulation.

        This function conducts a simulation, extracts the spiking activity of excitatory neurons, and generates a raster plot to visualize the timing of spikes. 
        The resulting raster plot is saved as both an image (SVG) and an interactive HTML plot.
        The generated raster plot is saved in the 'plots' and 'plots_html' directories, and the file names include the value of 'p' in their names.

        Usage:
        Call this function to generate a raster plot of excitatory neuron spiking activity and save it as an image and an interactive HTML file 
        in the 'plots' and 'plots_html' directories, respectively.

        Notes:
        - The simulation is run internally using the 'run_simulation' function.
        - 'v': The membrane potential activity of the network during the simulation.
        - 'e_neurons': Number of excitatory neurons per module.
        - 'nb_modules': Number of modules in the network.
        - 'p': The rewiring probability for excitatory to excitatory connections.

        """
        
        self.run_simulation()
        V = self.v
        # select only excitatory neurons
        V = V[:, :self.e_neurons*self.nb_modules]
        t, n = np.where(V > 29)
        
        fig = px.scatter(x=t, y=n)
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(xaxis_title="Time (ms)", yaxis_title="Neuron index")
        # remove grey background
        fig.update_layout(plot_bgcolor='white')
        # add border layout in black
        fig.update_xaxes(showline = True, linecolor = 'black', linewidth = 1, row = 1, col = 1, mirror = True)
        fig.update_yaxes(showline = True, linecolor = 'black', linewidth = 1, row = 1, col = 1, mirror = True)
        # fig.show()
        fig.write_image(f"plots/raster_p{self.p}.svg")
        fig.write_html(f"plots_html/raster_p{self.p}.html")

    def plot_firing_rate(self):
        """ Create a plot of mean firing rates in network modules during a simulation.

        This function calculates the mean firing rates for each module during a simulation, and then plots these rates over time. 
        The plot is saved as both an image (SVG) and an interactive HTML plot.
        The generated plot displays mean firing rates over time for each module and is saved in the 'plots' and 'plots_html' directories, 
        with the value of 'p' included in the file names.

        Usage:
        Call this function to calculate and visualize the mean firing rates of network modules over time, and save the plot 
        as an image and an interactive HTML file in the 'plots' and 'plots_html' directories, respectively.

        Notes:
        - The mean firing rates are computed using the 'compute_mean_firing_rates' function.
        - 'firing_rates': List of mean firing rates for each module.
        - 'p': The rewiring probability for excitatory to excitatory connections.

        """
        
        self.compute_mean_firing_rates()
        fig = go.Figure()
        for i in range(8):
            # add line corresponding to self.firing_rates[i]
            x_values = [j*20 for j in range(len(self.firing_rates[i]))]
            fig.add_trace(go.Scatter(x=x_values, y=self.firing_rates[i],
                    mode='lines',
                    name=f'{i}'))
        fig.update_layout(legend_title_text='Modules')
        fig.update_layout(xaxis_title="Time (ms)", yaxis_title="Mean firing rate")
        # remove grey background
        fig.update_layout(plot_bgcolor='white')
        # add border layout in black
        fig.update_xaxes(showline = True, linecolor = 'black', linewidth = 1,  mirror = True)
        fig.update_yaxes(showline = True, linecolor = 'black', linewidth = 1, mirror = True)
        # fig.show()
        fig.write_image(f"plots/firing_rate_p{self.p}.svg")
        fig.write_html(f"plots_html/firing_rate_p{self.p}.html")
        
# if main, run this code
if __name__ == "__main__":
    
    nb_modules = 8
    e_neurons = 100
    i_neurons = 200
    nb_connections = 1000
    time_simulation = 1000

    probabilities = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    for p in probabilities:
        print(p)
        net = SWMNetwork(nb_modules, e_neurons, i_neurons, nb_connections, p, time_simulation)
        # a) Generate a plot of the matrix connectivity
        net.plot_connectivity()
        # b) Generate a raster plot of the neuron firing in a 1000ms run.
        net.plot_raster()
        # c) Generate a plot of the mean firing rate in each of the eight modules for a 1000ms run.
        net.plot_firing_rate()
        

