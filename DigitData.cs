using Microsoft.ML.Data;

public class DigitData
{
    [LoadColumn(0)]
    public float Label { get; set; }

    [LoadColumn(1, 784)]
    [VectorType(784)]
    public float[]? PixelValues { get; set; }
}
