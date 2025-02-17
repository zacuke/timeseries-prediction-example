﻿using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using static Tensorflow.KerasApi;
namespace timeseries_prediction_example;

public class RnnModel : ModelBase, ITimeSeriesTask
{
    protected override Model BuildModel()
    {
        var layers = new List<ILayer>
        {
            keras.layers.LSTM(32, return_sequences:true),
            keras.layers.Dense(1)
        };
        var model = keras.Sequential(layers);
        model.compile(loss: keras.losses.MeanSquaredError(), optimizer: keras.optimizers.Adam(), metrics: new string[1] { "mae" });

        return model;

    }

    public void model_summary()
    {
        var rnn = BuildModel();
        rnn.summary();
    }

}
