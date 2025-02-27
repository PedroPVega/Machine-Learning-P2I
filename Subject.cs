public class Subject
{
    private static int NumberOfChannels = 19;
    private static int NumberOfLocalMaxima = 10;
    private static int BarycentersId = -1;
    private static Random rdn = new Random();
    public int Id {get;private set;}
    public double[,,] SubjectData {get; set;}

    public Subject()
    {
        Id = BarycentersId--;
        SubjectData = new double[NumberOfChannels, 2, NumberOfLocalMaxima];
        SetSubCoords2Zero();
    }

    public Subject(Subject s) // meant for replacing barycenters with no points, create new random Barycenter
    {
        Id = s.Id;
        SubjectData = new double[s.SubjectData.GetLength(0), 2, s.SubjectData.GetLength(2)];
        for (int i = 0; i < this.SubjectData.GetLength(0); i++)
        {
            for (int j = 0; j < this.SubjectData.GetLength(2); j++)
            {
                this.SubjectData[i,0,j] = rdn.NextDouble() * (230 - 10) + 10;
                this.SubjectData[i,1,j] = rdn.NextDouble() * (20 + 15) - 15;
            }
        }
    }
    public Subject(string csvFile)
    {
        // Defining value of subject's id
        int id;
        id = Convert.ToInt32(csvFile.Substring(csvFile.Length-7,3));
        SetId(id);
        // Defining the subject's eeg processed data
        SubjectData = new double[NumberOfChannels,2,NumberOfLocalMaxima]; // 19 electrodes, frequencies and power, 10 local maxima
        FillInData(csvFile);
    }

    public Subject(double[] minFreqs, double[] maxFreqs, double[] minPower, double[] maxPower)
    {
        // Constructeur utilisé pour faire un barycentre
        // On utilise la classe Subject utilisée pour stocker les données des sujets
        // On créé un "sujet artificiel" qui nous servira de barycentre pour le k means
        Id = BarycentersId--;
        SubjectData = new double[NumberOfChannels,2,NumberOfLocalMaxima];
        double minimum;
        for (int i = 0; i < 19; i++)
        {
            for (int t = 0; t < 10; t++)
            {
                minimum = minFreqs[t];
                SubjectData[i,0,t] = 1.00;
                SubjectData[i,0,t] = rdn.NextDouble() * (maxFreqs[t] - minimum) + minimum;
                    
                minimum = minPower[t];
                SubjectData[i,1,t] = rdn.NextDouble() * (maxPower[t] - minimum) + minimum;           
            }
        }
    }

    public void SetSubCoords2Zero()
    {
        for (int i = 0; i < this.SubjectData.GetLength(0); i++)
        {
            for (int j = 0; j < this.SubjectData.GetLength(2); j++)
            {
                this.SubjectData[i,0,j] = 0;
                this.SubjectData[i,1,j] = 0;
            }
        }
    }

    public void SetId(int n)
    {
        Id = n;
    }

    public override string ToString()
    {
        return base.ToString() + ' ' + Convert.ToString(Id);
    }

    public void PrintCoords()
    {
        Console.WriteLine("-----------------------------------------------------------------");
        for (int i = 0; i < NumberOfChannels; i++)
        {
            Console.WriteLine("Electrode {0}",i+1);
            for (int j = 0; j < NumberOfLocalMaxima; j++)
            {
                Console.WriteLine("Freq{0} = {1}   ; Pow{0} = {2}",j+1, SubjectData[i,0,j], SubjectData[i,1,j]);
            }
        }
        Console.WriteLine();
    }

    private void FillInData(string file) // If debbuging and doesn't work in unity, try : dotnet add package CsvHelper
    {
        // Console.WriteLine("Filling in data of subject {0}",Id);
        // -----------------------------------------------------------------------------
        string filePath = "./01_Training_Set/"+file;
        using (StreamReader reader = new StreamReader(filePath))
        {
            reader.ReadLine(); // First line of no interest to us
            string[]? values;
            string? line = reader.ReadLine();  // Read 2nd line
            values = line.Split(','); // Split by comma
            for (int j = 0; j < NumberOfChannels; j++) // 19 channels
            {
                for (int t = 0; t < NumberOfLocalMaxima; t++) // 10 local maxima
                {
                    SubjectData[j,0,t] = Convert.ToInt32(values[j*10 + t]);                        
                }
                    
            }

            line = reader.ReadLine();  // Read 3rd line
            values = line.Split(','); // Split by comma
                
            for (int j = 0; j < NumberOfChannels; j++) // 19 channels
            {
                for (int t = 0; t < NumberOfLocalMaxima; t++) // 10 local maxima
                {
                    SubjectData[j,1,t] = Convert.ToDouble(values[j*10 + t]);                        
                }
            }
            line = reader.ReadLine(); // Fermer le streamline
            // if (reader.EndOfStream)
                // Console.WriteLine("All data for subject {0} has been loaded", Id);
        }
        // -----------------------------------------------------------------------------
    }

    
}