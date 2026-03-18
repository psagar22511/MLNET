using Microsoft.ML.Data;

namespace MLAPI.Model
{
    public class EmailData
    {
        [LoadColumn(0)]
        public bool Label { get; set; } // 1 = positive, 0 = negative
        [LoadColumn(1)]
        public string Text { get; set; } = string.Empty;
    }
}
