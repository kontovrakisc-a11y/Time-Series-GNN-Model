#  WeatherGNN: Scientist-in-the-Loop Spatial-Temporal Forecasting




##  1. The Forecasting Challenge & Core Idea
Meteorological forecasting is a notoriously complex problem governed by non-linear fluid dynamics and thermodynamics. Traditional deep learning approaches (like LSTMs or standard CNNs) excel at finding patterns over time, but they treat every weather station as an isolated island. 

In reality, weather is dictated by **Spatial Advection**—a cold front hitting Station A will physically travel across the map and hit Station B a few hours later. If a neural network doesn't know the geographic and topographical relationship between Station A and Station B, it is forecasting blind.

**The Solution:** This project bridges the gap between deep learning and physical meteorology by introducing a **"Scientist-in-the-Loop" Graph Neural Network (GNN)**. We use Vision Large Language Models (LLMs) to analyze satellite imagery and explicitly teach the neural network how weather moves across the Californian landscape.

---

##  2. Data Acquisition: CIMIS & NASA
### The Nodes: CIMIS Weather Stations
We collected raw, hourly meteorological data from **41 active weather stations** managed by the California Irrigation Management Information System (CIMIS). 
* **Features Collected:** Temperature, Dew Point, Humidity, Precipitation, Solar Radiation, Soil Temperature, Wind Speed, Wind Direction, and Evapotranspiration (ETo).
* **Timeframe:** Thousands of consecutive hours of data.

### The Edges: NASA Topographical Metadata
Weather doesn't travel in a straight vacuum. A mountain range between two stations will block wind, while a flat valley will accelerate it. To capture this, we queried **NASA Satellite Metadata** to extract the elevation profiles and surface roughness characteristics of the exact geographical coordinates lying *between* the stations.

---

##  3. "Scientist-in-the-Loop" Graph Creation
Instead of relying on a naive Euclidean distance formula to connect our weather stations in a graph, we designed a pipeline that replicates the thought process of a human meteorologist.

1. **Visual Context Gathering**
   For every weather station, we scraped two sets of images via Google Earth:
   * **Close-Up Imagery**: To capture local surface roughness (e.g., concrete urban heat islands vs. grassy agricultural fields).
   * **Broader Satellite Imagery**: To capture the macro-topography (e.g., surrounding mountains, valleys, and coastlines).

2. **LLM Image-to-Text Interpretation**
   We passed these images into a Vision LLM with a highly specific prompt:
   > *"Act as an expert meteorologist and aerodynamicist. Analyze these images and describe the aerodynamic drag, surface roughness, and topographical wind-flow characteristics of this location."*

3. **Generating Learnable Edge Weights**
   For every pair of stations (Node A and Node B), we concatenated:
   * The LLM's physical description of Station A.
   * The LLM's physical description of Station B.
   * The NASA metadata describing the terrain between them.
   
   This combined text was fed back into the LLM, prompting it to output an **Aerodynamic Importance Score** (a numerical weight estimating how easily weather flows from A to B). These scores were formatted into an explicit PyTorch Geometric `edge_index`.

---

##  4. Data Engineering & Cleansing
Rigorous time-series forecasting requires bulletproof data integrity to prevent data leakage.
* **Imputation:** Linear interpolation was used to handle occasional sensor dropouts.
* **The Data Cube:** The tabular data was stacked into a massive 3D Spatio-Temporal tensor: `[41 Nodes, 8832 Hours, 9 Features]`.
* **Strict Normalization:** Z-Score normalization (Mean and Standard Deviation) was calculated **exclusively on the training slice** (`[:train_size]`). Applying normalization across the entire dataset is a common trap that leaks future test distributions into the training phase; our pipeline strictly prevents this.

---

##  5. Model Architectures
We predict the weather at `t+48` based on a 48-hour historical window (`t` to `t+47`).

### Baseline: The `TemporalCNN`
To prove the GNN's worth, we first built a brutally competitive 1D-CNN baseline using a physical dual-path design:
* **Pulse Path:** A Convolutional + Linear layer path designed to capture high-frequency volatility (e.g., sudden wind gusts).
* **Trend Path:** An Adaptive Average Pooling path designed to capture low-frequency mean states (e.g., the underlying daily temperature curve).

### Advanced Model: The `Balanced_HSTGNN`
This model shares the exact same dual-path temporal encoder, but adds a crucial spatial component:
* **Learnable Spatial Weights:** The LLM's Aerodynamic Importance Scores are loaded into the model as an `nn.Parameter`. This allows the network to use backpropagation to physically optimize and fine-tune the spatial advection matrix and the LLM's aerodynamic assumptions!
* **Gated Graph Convolutions:** The node embeddings are passed through a `GatedGraphConv`, allowing the stations to pass "weather messages" to their neighbors.
* **Residual Connections:** We add the original temporal features back into the spatial features to prevent the graph from over-smoothing localized weather events.

---

##  6. High-Level Results
We ran a Random Search over a hyperparameter grid (Hidden Dimensions, Learning Rates, GNN Layers) paired with Early Stopping. The models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-Squared ($R^2$), and implicitly compared against a **Naive Persistence Baseline** (which assumes tomorrow's weather will be identical to today's) before directly comparing the pure `TemporalCNN` and the graph-enhanced `Balanced_HSTGNN`.

**Key Findings:**

### 1. Comparison to the Naive Baseline
Both the `TemporalCNN` and the `Balanced_HSTGNN` deep learning architectures outperformed the Naive Persistence Baseline. This indicates that both models learned genuine meteorological patterns rather than simply predicting the recent past.

### 2. GNN-CNN vs. Pure CNN
While the pure CNN served as a strong temporal baseline, adding the learnable Spatial Graph (`Balanced_HSTGNN`) yielded measurable error reductions across the majority of the meteorological variables.

#### Final Results Matrix (Physical Units)

| Variable | CNN MAE | GNN MAE | CNN RMSE | GNN RMSE | CNN R² | GNN R² | MAE Improv. (%) | RMSE Improv. (%) | R² Improv. (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Temp_C** | 0.5703 | 0.5739 | 0.7581 | 0.7637 | 0.9873 | 0.9872 | -0.63% | -0.74% | -0.02% |
| **DewPoint_C** | 0.5290 | 0.5154 | 0.7598 | 0.7484 | 0.9691 | 0.9701 | 2.57% | 1.51% | 0.10% |
| **Humidity_pct** | 2.4857 | 2.4248 | 3.4842 | 3.4126 | 0.9734 | 0.9745 | 2.45% | 2.06% | 0.11% |
| **Precip_mm** | 0.0140 | 0.0129 | 0.0853 | 0.0882 | 0.2937 | 0.2443 | 7.81% | -3.44% | -16.82% |
| **SolarRad_Wm2** | 26.5255 | 24.4198 | 42.1697 | 40.6150 | 0.9868 | 0.9877 | 7.94% | 3.69% | 0.10% |
| **SoilTemp_C** | 0.2862 | 0.2723 | 0.3646 | 0.3556 | 0.9910 | 0.9914 | 4.87% | 2.48% | 0.04% |
| **Wind_ms** | 0.3194 | 0.3091 | 0.4351 | 0.4219 | 0.8939 | 0.9003 | 3.20% | 3.05% | 0.71% |
| **WindDir_deg** | 36.5329 | 36.3162 | 61.6294 | 61.6034 | 0.5297 | 0.5301 | 0.59% | 0.04% | 0.07% |
| **ETo_mm** | 0.0204 | 0.0189 | 0.0317 | 0.0296 | 0.9868 | 0.9885 | 7.65% | 6.76% | 0.17% |

* **What Improved:** Introducing the learnable Spatial Graph yielded consistent performance gains in variables heavily influenced by physical spatial flow and advection:
   * **Solar Radiation:** ~7.9% improvement in MAE
   * **Precipitation:** ~7.8% improvement in MAE
   * **Evapotranspiration (ETo):** ~7.6% improvement in MAE
   * **Soil Temperature:** ~4.8% improvement in MAE
   * **Wind Speed:** ~3.2% improvement in MAE
   * **Humidity & Dew Point:** ~2.4% and ~2.5% improvements in MAE
* **What Did Not Improve:** 
   * **Temperature (`Temp_C`):** The pure `TemporalCNN` baseline slightly outperformed the GNN on baseline Temperature (by ~0.6% in MAE). This suggests that localized temporal trends (like the daily cyclical warming and cooling of the sun) are a much stronger predictor of generic temperature than spatial movement.
   * **Precipitation RMSE and R²:** While the GNN had a significantly better absolute error (MAE) for Precipitation, it suffered in RMSE and R² compared to the CNN. Because precipitation is an inherently sparse, highly non-linear, and "spiky" variable, predicting exact peaks spatially remains incredibly challenging.

**Conclusion:**
By allowing the network to "see" incoming weather fronts via the Graph Neural Network, we successfully proved that Geography and Topography are indispensable features in modern meteorological forecasting, particularly for spatially dynamic features like Solar Radiation, Wind, and Evapotranspiration.
