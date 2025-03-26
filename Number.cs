public class Number
{
    public static int ImgSize = 784;
    public int[] Pixels {get; set;}    
    private static int BarycentersId = 0;
    private static int NumbersId = 1;
    private static Random rdn = new Random();
    public int Id {get;private set;}
    public int Label {get; set;}
    public int DesignatedClass {get; set;}

    public Number(int[]? vector = null, bool zeros = false, bool barycenter = false, bool random = false)
    {
        DesignatedClass = -1; // no designated class at the start
        Pixels = new int[ImgSize]; // image of 28*28 pixels
        Label = -1;
        if (vector != null)
        {
            if(vector.Length == ImgSize)
            {
                Label = -1;
                FillInData(vector, false);
            }
            else
            {
                Label = vector[0];
                FillInData(vector, true);
            }
        }
        if(zeros) // fill with zeros
        {
            Set2Empty();
        }
        if (barycenter) // initialise as barycenter
        {
            Id = BarycentersId--;
        }
        else
        {
            Id = NumbersId++;
        }
        if (random) // creation du nombre à l'envers
        {
            for (int i = 0; i < ImgSize; i++)
            {
                Pixels[i] = rdn.Next(256); 
            }
        }
    }

    public void FillInData(int[] vector, bool extractingData)
    {

        for (int i = 0; i < Pixels.Length; i++)
        {
            Pixels[i] = vector[i+ Convert.ToInt32(extractingData)];
        }
    }

    public void Set2Empty()
    {
        for (int i = 0; i < Pixels.Length; i++)
        {
            Pixels[i] = 0;
        }
    }

    public override string ToString()
    {
        return base.ToString() + ' ' + Convert.ToString(Label);
    }

    public void PrintCoords()
    {
        Console.WriteLine("-----------------------------------------------------------------");
        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++)
            {
                Console.Write("{0},",Pixels[i*28 + j]);
            }
            Console.WriteLine();
        }
        Console.WriteLine();
    }

    public void ResetBarsIds()
    {
        BarycentersId = 0;
    }

}