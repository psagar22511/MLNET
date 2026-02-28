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

        string dataPath = @"C:\Sagar-PC\Learning\MLNET\NewsCategoryClassification\NewsCategory.csv";
        string modelPath = @"C:\Sagar-PC\Learning\MLNET\NewsCategoryClassification\newsCategoryClassification.zip";

        new NewsCategoryClassficationService(dataPath, modelPath);
    }
}