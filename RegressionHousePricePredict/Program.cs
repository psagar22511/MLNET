

using RegressionHousePricePredict;

class Program
{
    static void Main(string[] args)
    {
        //Regression Classification Example

        string dataPath = @"C:\Sagar-PC\Learning\ML\RegressionHousePricePredict\Housing.csv";
        string modelPath = @"C:\Sagar-PC\Learning\ML\RegressionHousePricePredict\housePricePredict.zip";

        var spamDetector = new HousePredictService(dataPath, modelPath);

        var prediction = spamDetector.Predict(1500, 3, 5);

        Console.WriteLine($"Predicted Price: {prediction.PredictedPrice}");
    }
}