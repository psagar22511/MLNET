using Microsoft.ML.Data;

namespace MLAPI.Model
{
    public class EmailData
    {
        [LoadColumn(0)]
        public bool Label { get; set; } // Spam or Not
        [LoadColumn(1)]
        public string Text { get; set; } = string.Empty;
    }
}
