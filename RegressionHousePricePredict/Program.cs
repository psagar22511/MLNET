using Microsoft.ML;

using RegressionHousePricePredict;

class Program
{
    static void Main(string[] args)
    {
        //Regression Classification Example (House price prediction (Price prediction))

        //string dataPath = @"C:\Sagar-PC\Learning\MLNET\RegressionHousePricePredict\Housing.csv";
        //string modelPath = @"C:\Sagar-PC\Learning\MLNET\RegressionHousePricePredict\housePricePredict.zip";

        string dataPath = @"C:\SagarPatel\Practice\ML.NET\RegressionHousePricePredict\Housing.csv";
        string modelPath = @"C:\SagarPatel\Practice\ML.NET\RegressionHousePricePredict\housePricePredict.zip";

        var spamDetector = new HousePredictService(dataPath, modelPath);

        var prediction = spamDetector.Predict(1500, 3, 5);

        Console.WriteLine($"Predicted Price: {prediction.PredictedPrice}");
    }
}