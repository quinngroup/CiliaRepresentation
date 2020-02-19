from NVP import NVP as model
#from NVP import Driver
import plotly.graph_objects as go


figure = go.scatterPlot(model.encoder(x))
figure.show
