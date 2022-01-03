using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace House_Prices_Prediction_MLNet
{
    class Program
    {
        public class HouseData
        {
            public float Size { get; set; }
            public float Price { get; set; }
        }

        public class Prediction
        {
            [ColumnName("Score")]
            public float Price { get; set; }
        }
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            //Import or create training data
            HouseData[] houseData =
            {
                new HouseData(){Size=1.1F,Price=1.2F},
                new HouseData(){Size=1.9F,Price=2.3F},
                new HouseData(){Size=2.8F,Price=3.0F},
                new HouseData(){Size=3.4F,Price=3.7F}
            };

            IDataView trainingData = mlContext.Data.LoadFromEnumerable(houseData);

            //Specify data preparation and model training pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "Size" })
                .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));

            //Train model
            var model = pipeline.Fit(trainingData);

            //Evaluate the model
            HouseData[] testHouseData =
            {
                new HouseData(){Size=1.1F,Price=0.98F},
                new HouseData(){Size=1.9F,Price=2.1F},
                new HouseData(){Size=2.8F,Price=2.9F},
                new HouseData(){Size=3.4F,Price=3.6F}
            };

            var testHouseDataView = mlContext.Data.LoadFromEnumerable(testHouseData);
            var testPriceDataView = model.Transform(testHouseDataView);

            var metrics = mlContext.Regression.Evaluate(testPriceDataView, labelColumnName: "Price");

            Console.WriteLine($"R^2: {metrics.RSquared:0.##}");
            Console.WriteLine($"RMS error: {metrics.RootMeanSquaredError:0.##}");

            //Make a prediction
            var size = new HouseData() { Size = 2.5F };
            var price = mlContext.Model.CreatePredictionEngine<HouseData, Prediction>(model).Predict(size);

            Console.WriteLine($"Predicted price for size: {size.Size * 1000} sq ft= {price.Price * 100:C}k");// Predicted price for size: 2500 sq ft= $261.98k

            Console.ReadLine();
        }
    }
}
