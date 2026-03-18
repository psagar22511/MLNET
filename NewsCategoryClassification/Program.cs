using NewsCategoryClassification;

class Program
{
    static void Main(string[] args)
    {
        //Multiclass Classification Example
        /* What You Learn Here:
        Label encoding
        Text feature engineering
        Multiclass algorithms
        Accuracy metrics */

        //string modelPath = @"C:\Sagar-PC\Learning\MLNET\RegressionHousePricePredict\housePricePredict.zip";
        //string modelPath = @"C:\Sagar-PC\Learning\MLNET\NewsCategoryClassification\newsCategoryClassification.zip";

        string dataPath = @"C:\SagarPatel\Practice\ML.NET\NewsCategoryClassification\NewsCategory.csv";
        string modelPath = @"C:\SagarPatel\Practice\ML.NET\NewsCategoryClassification\newsCategoryClassification.zip";

        new NewsCategoryClassficationService(dataPath, modelPath);
    }
}