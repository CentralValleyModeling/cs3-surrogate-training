# EC_X2_Surrogate
This code has been written in Python, adapted from an ANN model originally developed in Matlab. A comprehensive version of this code has been developed by Nicky Sandhu is available on GitHub (https://github.com/dwr-psandhu/ann_calsim). This updated version is based on preprocessed input data and incorporates the normalization process directly into the model building phase utilizing the TensorFlow library. As a result, the trained model can be exported in .pb format, enabling seamless integration and utilization in other programming environments, including Java within Eclipse. The reference study for the original model: Jayasundara, N. C., Seneviratne, S. A., Reyes, E., & Chung, F. I. (2020). Artificial neural network for Sacramento–San Joaquin Delta flow–salinity relationship for CalSim 3.0. Journal of Water Resources Planning and Management, 146(4), 04020015. To run and test the code, first, establish the necessary environment by executing "conda env create -f environment.yml" in your Conda prompt. Once set up, you can execute train.ipynb using Jupyter Notebook.

```mermaid
flowchart TB

%% =================================================
%% TOP: ANN Training & CalSim Connectivity
%% =================================================
subgraph TOP["Model Training & Deployment"]
  direction LR

  subgraph ANNTRAIN["ANN Training"]
    direction LR
    CAS["ANN Training"] --> TF["TensorFlow SavedModel"]
  end

  subgraph APP["CalSim Application"]
    direction LR
    CS["CalSim"] <--> SURR["calsurrogate Java"]
  end

  TF --> SURR
end


%% =================================================
%% BOTTOM: Full ANN Input → EC Estimation Pipeline
%% =================================================
subgraph BOTTOM["ANN-Based EC Estimation Framework"]
  direction LR

  %% -------------------------
  %% Inputs
  %% -------------------------
  subgraph HYDRO["Hydrologic & Operational Inputs"]
    direction TB
    X1["Delta Cross Channel Operation"]
    X2["Export"]
    X3["Northern Flow"]
    X4["San Joaquin Flow"]
    X5["Astro Planning Tide"]
    X6["Delta Channel Depletion"]
    X7["Suisun Marsh Salinity Control Gate"]
  end

  %% -------------------------
  %% Pre-processing
  %% -------------------------
  subgraph PRE["Pre-processing"]
    direction LR
    HIST["Antecedent history<br/>18 values per variable"]
    FLAT["Flattened input vector<br/>7 × 18 = 126"]
    HIST --> FLAT
  end

  %% -------------------------
  %% ANN Model
  %% -------------------------
  subgraph NET["ANN Model Structure (TensorFlow)"]
    direction LR
    IN["Input layer<br/>126 nodes"]
    H1["1st hidden layer<br/>8 nodes<br/>(Sigmoid)"]
    H2["2nd hidden layer<br/>2 nodes<br/>(Sigmoid)"]
    OUT["Output layer<br/>1 node<br/>(Linear / ReLU)"]
    IN --> H1 --> H2 --> OUT
  end

  EC["Estimated EC"]

  %% -------------------------
  %% Connections
  %% -------------------------
  X1 --> HIST
  X2 --> HIST
  X3 --> HIST
  X4 --> HIST
  X5 --> HIST
  X6 --> HIST
  X7 --> HIST

  FLAT --> IN
  OUT --> EC
end
