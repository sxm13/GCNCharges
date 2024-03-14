### Overall workflow
![Workflow of this work](../figs/workflow.png "workflow")

# Structure of repository
```
.
├── ..
├── cif2data.py                                         # covert cif to data and 
│   ├── check_all                                       # Notebook where MOFs are used to create corresponding graph and graph features
│   ├── ddec_xtals                                      # Crystal structures with DDEC charges assigned to them{1}
│   ├── graphs                                          # Directory with graph information (edges, node features and node labels) for the DDEC set
│   ├── CoRE_v2_ASR                                     # Crystal structure from CoRE v2 (All Solvents Removed){2}
│   ├── CoRE_v2_FSR                                     # Crystal structure from CoRE v2 (Free Solvents Removed){2}
│   ├── deployment_graphs_ASR                           # Directory with graph information for the CoRE_v2 (All Solvents Removed) set
│   ├── deployment_graphs_FSR                           # Directory with graph information for the CoRE_v2 (Free Solvents Removed) set
│   └── Bonds.jl                                        # Bond generation code. Utilized in graph creation notebook
│
├── MPNN                                                # Directory containing the Message Passing Neural Network (MPNN) code and results
│   ├── model.py                                        # Contains the neural network model 
│   ├── charge_prediction_system.py                     # For training the model
│   ├── data_handling.py                                # Reads in graphs from [../build_graphs/graphs] and  returns a datalist
│   ├── main.py                                         # Loads in datalist from [./data_handling.py]. Split it into training, validation and testing dataset. Uses [./charge_prediction_system] for training the model [./model.py] and tests it
│   ├── main.ipynb                                      # Notebook for main.py
│   └── results                                         # Contains results from the MPNN
│       ├── embedding                                   # Element embedding from the MPNN
│       └── graphs                                     # Different graphs related to training and testing
│
├── embedding_visualization                             # Element Embedding visualizations
│   └── Embedding_Visualization.ipynb                   # Notebook for element embedding visualization. Utilizes UMAP, t-SNE and PCA
│
├── deployment                                          # Code for deployment dataset, where MPNN charges are assigned to the CoRE v2{2} dataset
│   ├── data_handling.py                                # Reads in graph information from [../build_graphs/deployment_graphs[A/F]SR] and generates a data list
│   ├── deployment_main.py                              # Main file for reading the graphs, loading the model and generating charge predictions for deployment sets
│   ├── deployment_main.ipynb                           # Notebook for deployment_main.py
│   ├── model.py                                        # Required by [./deployment_main.py/ipynb] to load the trained model [./models_deployment.pt]  
│   └── results                                         # Results of charge predictions for the deployment sets
│       └── predictions                                 # Charge predictions
│           ├── deployment_graphs_ASR                   # - for CoRE_v2_ASR
│           └── deployment_graphs_FSR                   # - for CoRE_v2_FSR
│
├── Charge_Assigned_CoRE_MOFs                           # CoRE v2 structures with MPNN charges assigned to them
│   ├── MPNN_CoRE-ASR.tar.gz                            # - CoRE v2 ASR (All Solvents Removed) structures with MPNN charges
│   └── MPNN_CoRE-FSR.tar.gz                            # - CoRE v2 FSR (Free Solvents Removed) structures with MPNN charges
│
└── adsorption_simulations                              # Adsorption simulation details for Henry coefficients
    ├── analyze_henry.ipynb                             # - Notebook that analyzes results stored in simulation results directory
    ├── run_henry.jl                                    # - The Julia script which runs the Henry coefficients
    ├── run_henry.sh                                    # - 
    ├── submit_henry.sh                                 # - Two files used to submit Adsorption calculations to a cluster
    ├── iqeq_xtals                                      # - Crystals (from a test set in one of our MPNN runs) with I-QEq charges assigned
    ├── mpnn_xtals                                      # - Crystals (from a test set in one of our MPNN runs) with MPNN charges assigned
    ├── ddec_xtals                                      # - Crystals (from a test set in one of our MPNN runs) with DDEC charges assigned
    └── simulations.tar.gz                              # - Simulations results stored in a tarball.
```
