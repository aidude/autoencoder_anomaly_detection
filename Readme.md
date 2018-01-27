	Anomaly detection with Autoencoders


Anomaly detection is an effective approach of dealing with problems in the area of network security. Rapid development in technology has raised the need for an effective detection system using machine learning in order to detect novel and advanced intrusions. At most present IDS try to perform their task in real time but their performance hinders as they undergo different level of analysis or their reaction to limit the damage of some intrusions by terminating the network connection, a real time is not always achieved. With increasing amount of data being transmitted day by day from one network to another, there is an increased need to identify intrusion in such large datasets effectively and in a timely manner and data mining and machine learning approaches could prove effective in this regard. 
Anomaly or Outlier detection is especially tricky problem in networks,financial transactions and real world data analysis. The training examples are so low in number its tough to assess the statistical properties of anomalies.

Autoencoders

High-dimensional data can be converted to low-dimensional codes by training a multilayer neural
network with a small central layer to reconstruct high-dimensional input vectors. Gradient descent
can be used for fine-tuning the weights in such ‘‘autoencoder’’ networks, but this works well only if
the initial weights are close to a good solution. We describe an effective way of initializing the
weights that allows deep autoencoder networks to learn low-dimensional codes that work much
better than principal components analysis as a tool to reduce the dimensionality of data.