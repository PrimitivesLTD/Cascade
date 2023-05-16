using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Forms;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Threading.Tasks;

using AForge.Imaging;
using CNTK;

using MessageBox = System.Windows.MessageBox;
using MouseEventHandler = System.Windows.Input.MouseEventHandler;
using Path = System.Windows.Shapes.Path;
using Point = System.Windows.Point;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.PixelFormats;
using System.Threading;
using System.Diagnostics;

namespace PrimitivesCascade
{
    public partial class MainWindow : System.Windows.Window
    {
        private bool isDragging = false;
        private bool newSelection;
        private Point startPosition;
        FileInfo[] sourceFiles;
        private DirectoryInfo sourceDirectory;
        private DirectoryInfo goodSamplesDirectory;
        private DirectoryInfo badSamplesDirectory;

        public MainWindow()
        {
            InitializeComponent();
            ImageCanvas.MouseLeftButtonDown += new MouseButtonEventHandler(SelectionStart);
            ImageCanvas.MouseMove += new MouseEventHandler(SelectionDraw);
            ImageCanvas.MouseLeftButtonUp += new MouseButtonEventHandler(SelectionStop);
            ImageCanvas.MouseRightButtonDown += new MouseButtonEventHandler(RemoveLastSelection);
            _originalButtonColor = (SolidColorBrush)Train.Background;
        }

        private void SelectionStart(object sender, MouseButtonEventArgs e)
        {
            if (!isDragging)
            {
                isDragging = true;
                newSelection = true;
                startPosition = e.GetPosition(ImageCanvas);
            }
        }

        private void SelectionStop(object sender, MouseButtonEventArgs e)
        {
            isDragging = false;
        }

        private void SelectionDraw(object sender, System.Windows.Input.MouseEventArgs e)
        {
            if (isDragging)
            {
                Point currentPosition = e.GetPosition(ImageCanvas);

                if (newSelection)
                {
                    ImageCanvas.Children.Add(new Path
                    {
                        Data = new RectangleGeometry(new Rect(startPosition, currentPosition)),
                        Stroke = System.Windows.Media.Brushes.Red,
                        StrokeThickness = 1
                    });
                    newSelection = false;
                }
                else
                {
                    Path selection = ImageCanvas.Children.OfType<Path>().Last();
                    selection.Data = new RectangleGeometry(new Rect(startPosition, currentPosition));
                }
            }
        }

        private void RemoveLastSelection(object sender, MouseButtonEventArgs e)
        {
            if (ImageCanvas.Children.OfType<Path>().Any())
            {
                ImageCanvas.Children.Remove(ImageCanvas.Children.OfType<Path>().Last());
            }
        }

        private void OpenSourceDir_Click(object sender, System.Windows.RoutedEventArgs e)
        {
            using (var folderDialog = new FolderBrowserDialog())
            {
                folderDialog.Description = "Выберите папку";
                DialogResult result = folderDialog.ShowDialog();
                if (result == System.Windows.Forms.DialogResult.OK && !string.IsNullOrWhiteSpace(folderDialog.SelectedPath))
                {
                    sourceDirectory = new DirectoryInfo(folderDialog.SelectedPath);
                    sourceFiles = sourceDirectory.GetFiles();
                    for (int i = 0; i < sourceFiles.Length; i++)
                    {
                        SourceImagesDir.Items.Add(sourceFiles[i].Name);
                    }
                }
            }
        }

        private void SourceImagesDir_SelectedItemChanged(object sender, RoutedPropertyChangedEventArgs<object> e)
        {
            if (e.NewValue != null)
            {
                FileInfo file = sourceFiles.FirstOrDefault(x => x.Name == e.NewValue.ToString());
                if (file != null)
                {
                    ImageSource img = new BitmapImage(new Uri(file.FullName));
                    Image.Source = img;
                    ImageCanvas.Children.Clear();
                }
            }
        }

        private void CloseSourceDir_Click(object sender, System.Windows.RoutedEventArgs e)
        {
            SourceImagesDir.Items.Clear();
            sourceFiles = null;
        }

        private void ClearSourceDir_Click(object sender, System.Windows.RoutedEventArgs e)
        {
            if (sourceFiles != null && sourceFiles.Length > 0)
            {
                foreach (FileInfo file in sourceFiles)
                {
                    file.Delete();
                }
                SourceImagesDir.Items.Clear();
                sourceFiles = null;
            }
        }

        private void OpenSaveSamplesDir_Click(object sender, System.Windows.RoutedEventArgs e)
        {
            using (var folderDialog = new FolderBrowserDialog())
            {
                folderDialog.Description = "Выберите папку";
                DialogResult result = folderDialog.ShowDialog();
                if (result == System.Windows.Forms.DialogResult.OK && !string.IsNullOrWhiteSpace(folderDialog.SelectedPath))
                {
                    SaveSamplesDir.Text = folderDialog.SelectedPath;
                    goodSamplesDirectory = Directory.CreateDirectory(System.IO.Path.Combine(SaveSamplesDir.Text, "Good"));
                    badSamplesDirectory = Directory.CreateDirectory(System.IO.Path.Combine(SaveSamplesDir.Text, "Bad"));

                    DirectoryInfo dir = new DirectoryInfo(goodSamplesDirectory.FullName);
                    GoodSamplesCount.Content = dir.GetFiles().Length;

                    dir = new DirectoryInfo(badSamplesDirectory.FullName);
                    BadSamplesCount.Content = dir.GetFiles().Length;
                }
            }
        }

        private void SaveSample_Click(object sender, System.Windows.RoutedEventArgs e)
        {
            if (ImageCanvas.Children.OfType<Path>().Count() == 0)
            {
                MessageBox.Show("Сначала выделите образец на изображении");
            }
            else if (!IsGoodSample.IsChecked.HasValue || !IsBadSample.IsChecked.HasValue)
            {
                MessageBox.Show("Выберите тип образца");
            }
            else if (string.IsNullOrEmpty(SaveSamplesDir.Text))
            {
                MessageBox.Show("Выберите папку для сохранения образцов");
            }
            else
            {
                foreach (Path sample in ImageCanvas.Children.OfType<Path>())
                {
                    System.Drawing.Rectangle sampleRect = new System.Drawing.Rectangle((int)sample.Data.Bounds.X, (int)sample.Data.Bounds.Y, (int)sample.Data.Bounds.Width, (int)sample.Data.Bounds.Height);
                    MemoryStream ms = new MemoryStream();
                    var encoder = new BmpBitmapEncoder();
                    encoder.Frames.Add(BitmapFrame.Create(Image.Source as BitmapSource));
                    encoder.Save(ms);
                    ms.Flush();
                    Bitmap bmpImage = new Bitmap(ms);
                    Bitmap bmpCrop = bmpImage.Clone(sampleRect, bmpImage.PixelFormat);
                    if (IsInGrayscale.IsChecked.HasValue && IsInGrayscale.IsChecked.Value)
                    {
                        bmpCrop = Extensions.CopyAsGrayscale(bmpCrop);
                    }
                    if (IsGoodSample.IsChecked.Value)
                    {
                        int cropNumber = goodSamplesDirectory.GetFiles().Length;
                        string savePath = System.IO.Path.Combine(goodSamplesDirectory.FullName, $"{cropNumber}.bmp");
                        bmpCrop.Save(savePath, ImageFormat.Bmp);
                        GoodSamplesCount.Content = cropNumber + 1;
                        if (IsMarkupCreating.IsChecked.HasValue && IsMarkupCreating.IsChecked.Value)
                        {
                            using (StreamWriter sw = new StreamWriter(System.IO.Path.Combine(goodSamplesDirectory.FullName, "Good.dat"), true))
                            {
                                sw.WriteLine($"Good\\{cropNumber}.bmp 1 0 0 {bmpCrop.Width} {bmpCrop.Height}");
                            }
                        }
                    }
                    else if (IsBadSample.IsChecked.Value)
                    {
                        int cropNumber = badSamplesDirectory.GetFiles().Length;
                        string savePath = System.IO.Path.Combine(badSamplesDirectory.FullName, $"{cropNumber}.bmp");
                        bmpCrop.Save(savePath, ImageFormat.Bmp);
                        BadSamplesCount.Content = cropNumber + 1;
                        if (IsMarkupCreating.IsChecked.HasValue && IsMarkupCreating.IsChecked.Value)
                        {
                            using (StreamWriter sw = new StreamWriter(System.IO.Path.Combine(badSamplesDirectory.FullName, "Bad.dat"), true))
                            {
                                sw.WriteLine($"{badSamplesDirectory.FullName}\\{cropNumber}.bmp");
                            }
                        }
                    }
                    ms.Dispose();
                }
                ImageCanvas.Children.Clear();
                MessageBox.Show("Все образцы сохранены");
            }
        }

        private void ConvertExistingGrayscale_Click(object sender, System.Windows.RoutedEventArgs e)
        {
            if (!IsGoodSample.IsChecked.HasValue || !IsBadSample.IsChecked.HasValue)
            {
                MessageBox.Show("Выберите тип образца");
            }
            else if (string.IsNullOrEmpty(SaveSamplesDir.Text))
            {
                MessageBox.Show("Выберите папку для сохранения образцов");
            }
            else
            {
                DirectoryInfo targetDirectory = IsGoodSample.IsChecked.Value ? goodSamplesDirectory : badSamplesDirectory;
                DirectoryInfo tempDirectory = Directory.CreateDirectory(System.IO.Path.Combine(SaveSamplesDir.Text, "Temp"));
                List<FileInfo> samples = new List<FileInfo>(targetDirectory.GetFiles());
                for (int i = 0; i < samples.Count; i++)
                {
                    Bitmap source = new Bitmap(samples[i].FullName);
                    Bitmap result = Extensions.CopyAsGrayscale(source);
                    source.Dispose();
                    string savePath = System.IO.Path.Combine(tempDirectory.FullName, $"{samples[i].Name}");
                    result.Save(savePath, ImageFormat.Bmp);
                    result.Dispose();
                }
                samples.Clear();
                targetDirectory.Delete(true);
                Directory.Move(tempDirectory.FullName, targetDirectory.FullName);
                MessageBox.Show("Все образцы преобразованы в оттенки серого");
            }
        }

        private void RenameSamples_Click(object sender, System.Windows.RoutedEventArgs e)
        {
            if (!IsGoodSample.IsChecked.HasValue || !IsBadSample.IsChecked.HasValue)
            {
                MessageBox.Show("Выберите тип образца");
            }
            else if (string.IsNullOrEmpty(SaveSamplesDir.Text))
            {
                MessageBox.Show("Выберите папку для сохранения образцов");
            }
            else
            {
                DirectoryInfo targetDirectory = IsGoodSample.IsChecked.Value ? goodSamplesDirectory : badSamplesDirectory;
                DirectoryInfo tempDirectory = Directory.CreateDirectory(System.IO.Path.Combine(SaveSamplesDir.Text, "Temp"));
                List<FileInfo> samples = new List<FileInfo>(targetDirectory.GetFiles());
                for (int i = 0; i < samples.Count; i++)
                {
                    string savePath = System.IO.Path.Combine(tempDirectory.FullName, $"{i}.bmp");
                    samples[i].CopyTo(savePath);
                }
                samples.Clear();
                targetDirectory.Delete(true);
                Directory.Move(tempDirectory.FullName, targetDirectory.FullName);
                MessageBox.Show("Все образцы переименованы в порядке");
            }
        }

        private void RewriteMarkup_Click(object sender, System.Windows.RoutedEventArgs e)
        {
            if (IsGoodSample.IsChecked.Value)
            {
                FileInfo[] samples = goodSamplesDirectory.GetFiles();
                string markupFilePath = System.IO.Path.Combine(goodSamplesDirectory.FullName, "Good.dat");
                if (File.Exists(markupFilePath))
                {
                    File.Delete(markupFilePath);
                }
                using (StreamWriter sw = new StreamWriter(markupFilePath, true))
                {
                    for (int i = 0; i < samples.Length; i++)
                    {
                        Bitmap image = new Bitmap(samples[i].FullName);
                        sw.WriteLine($"Good\\{samples[i].Name} 1 0 0 {image.Width} {image.Height}");
                        image.Dispose();
                    }
                }
            }
            else if (IsBadSample.IsChecked.Value)
            {
                FileInfo[] samples = badSamplesDirectory.GetFiles();
                string markupFilePath = System.IO.Path.Combine(badSamplesDirectory.FullName, "Bad.dat");
                if (File.Exists(markupFilePath))
                {
                    File.Delete(markupFilePath);
                }
                using (StreamWriter sw = new StreamWriter(markupFilePath, true))
                {
                    for (int i = 0; i < samples.Length; i++)
                    {
                        sw.WriteLine($"{badSamplesDirectory.FullName}\\{samples[i].Name}");
                    }
                }
            }
            MessageBox.Show("Файл разметки был перезаписан");
        }

        private static float[] PreprocessImage(string imagePath, int width, int height)
        {
            try
            {
                using (Image<Rgba32> image = SixLabors.ImageSharp.Image.Load<Rgba32>(imagePath))
                {
                    image.Mutate(ctx => ctx.Resize(new ResizeOptions
                    {
                        Size = new SixLabors.ImageSharp.Size(width, height),
                        Sampler = KnownResamplers.Lanczos3
                    }));

                    var preprocessedImage = new float[width * height * 3];
                    for (int y = 0; y < height; y++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            var color = image[x, y];

                            // Normalize pixel values to [0, 1]
                            preprocessedImage[y * width + x] = color.R / 255.0f;
                            preprocessedImage[width * height + y * width + x] = color.G / 255.0f;
                            preprocessedImage[2 * width * height + y * width + x] = color.B / 255.0f;
                        }
                    }

                    return preprocessedImage;
                }
            }
            catch (Exception ex)
            {
                // Log error or simply ignore
                Console.WriteLine($"Исключение: {ex.Message}");
                return null;
            }
        }

        private static List<(float[] Image, float Label)> LoadBatchData(string[] imagePaths, float[] labels, int width, int height, int start, int count)
        {
            var batchData = new List<(float[] Image, float Label)>();
            for (int i = start; i < Math.Min(start + count, imagePaths.Length); i++)
            {
                var preprocessedImage = PreprocessImage(imagePaths[i], width, height);
                if (preprocessedImage != null)
                {
                    batchData.Add((preprocessedImage, labels[i]));
                }
            }
            return batchData;
        }

        public static Function ConvLayer(Variable input, int numFilters, int[] filterSize, DeviceDescriptor device, int stride = 1, bool padding = true)
        {
            int numInputChannels = input.Shape[input.Shape.Rank - 1];

            var convParams = new Parameter(new int[] { filterSize[0], filterSize[1], numInputChannels, numFilters },
                DataType.Float, CNTKLib.GlorotUniformInitializer(CNTKLib.DefaultParamInitScale, CNTKLib.SentinelValueForInferParamInitRank, CNTKLib.SentinelValueForInferParamInitRank, 1),
                device);

            var convFunction = CNTKLib.Convolution(convParams, input, new int[] { stride, stride, numInputChannels }, new BoolVector(new bool[] { padding, padding, false }));

            return convFunction;
        }

        public static Function ResNetBlock(Function input, int numFilters, DeviceDescriptor device, int stride = 1)
        {
            // First convolutional layer
            var conv1 = ConvLayer(input, numFilters, new int[] { 3, 3 }, device, stride);

            // Non-linearity
            var relu1 = CNTKLib.ReLU(conv1);

            // Second convolutional layer
            var conv2 = ConvLayer(relu1, numFilters, new int[] { 3, 3 }, device);

            // If stride is not 1, or input and output number of filters does not match, then we need to
            // adjust the input accordingly
            Function inputAdjusted;

            if (stride != 1 || input.Output.Shape[0] != numFilters)
            {
                inputAdjusted = ConvLayer(input, numFilters, new int[] { 1, 1 }, device, stride);
            }
            else
            {
                inputAdjusted = input;
            }

            // Addition
            var add = CNTKLib.Plus(conv2, inputAdjusted);

            // Final non-linearity
            var relu2 = CNTKLib.ReLU(add);

            return relu2;
        }
        private void OpenModelsFolder()
        {
            string modelsDir = System.IO.Path.Combine(baseDir, "Models");

            try
            {
                Process.Start("explorer.exe", modelsDir);
            }
            catch (Exception ex)
            {
                // Error handling, if failed to open the folder
                Console.WriteLine("Проблема с Проводником: " + ex.Message);
            }
        }

        private const int BatchSize = 16;  // You can adjust this depending on your memory availability

        private CancellationTokenSource _cancellationTokenSource;
        private SolidColorBrush _originalButtonColor;
        private bool _isTraining = false;
        string baseDir;
        private async void Train_Click(object sender, System.Windows.RoutedEventArgs e)
        {
            if (!_isTraining)
            {
                _originalButtonColor = (SolidColorBrush)Train.Background;
                baseDir = $@"{SaveSamplesDir.Text}";

                // We start training
                _isTraining = true;
                Train.Background = new SolidColorBrush(Colors.Orange); // Changing the button color
                DisableUI();
                _cancellationTokenSource = new CancellationTokenSource();

                try
                {
                    // We start training in a separate thread
                    await Task.Run(() =>
                    {
                        DeviceDescriptor device = DeviceDescriptor.CPUDevice;

                        // Load data
                        string[] goodImagePaths = Directory.GetFiles(System.IO.Path.Combine(baseDir, "Good"));
                        string[] badImagePaths = Directory.GetFiles(System.IO.Path.Combine(baseDir, "Bad"));
                        int width = 224, height = 224;

                        // Splitting the data into training and testing
                        Random rng = new Random();
                        string[] allImagePaths = goodImagePaths.Concat(badImagePaths).OrderBy(x => rng.Next()).ToArray();
                        float[] allLabels = goodImagePaths.Select(x => 1f).Concat(badImagePaths.Select(x => 0f)).OrderBy(x => rng.Next()).ToArray();

                        int testCount = allImagePaths.Length / 5;  // 20% of the data for testing
                        string[] testImagePaths = allImagePaths.Take(testCount).ToArray();
                        float[] testLabels = allLabels.Take(testCount).ToArray();
                        string[] trainImagePaths = allImagePaths.Skip(testCount).ToArray();
                        float[] trainLabels = allLabels.Skip(testCount).ToArray();

                        // Define total number of batches
                        int trainBatchCount = (int)Math.Ceiling((double)trainImagePaths.Length / BatchSize);

                        // Define network
                        int numOutputClasses = 1;  // Binary classification

                        var imageInput = Variable.InputVariable(new int[] { width, height, 3 }, DataType.Float);

                        // The first convolutional layer
                        var convInit = ConvLayer(imageInput, 64, new int[] { 7, 7 }, device, stride: 2);

                        // Max pooling layer
                        var poolInit = CNTKLib.Pooling(convInit, PoolingType.Max, new int[] { 3, 3 }, new int[] { 2, 2 });

                        // ResNet blocks
                        var block1 = ResNetBlock(poolInit, 64, device, 2);
                        var block2 = ResNetBlock(block1, 128, device, 2);
                        var block3 = ResNetBlock(block2, 256, device, 2);
                        var block4 = ResNetBlock(block3, 512, device, 2);

                        // Average pooling
                        var avgPool = CNTKLib.Pooling(block4, PoolingType.Average, block4.Output.Shape.Dimensions.Take(2).ToArray(), new int[] { 1, 1 });

                        // Fully connected layer
                        var Wfc = new Parameter(new int[] { 1, 1, NDShape.InferredDimension, numOutputClasses }, DataType.Float, CNTKLib.GlorotUniformInitializer(), device);
                        var Bfc = new Parameter(new int[] { numOutputClasses }, DataType.Float, 0, device);
                        var model = CNTKLib.Sigmoid(CNTKLib.Plus(Bfc, CNTKLib.Convolution(Wfc, avgPool)));

                        var labelInput = Variable.InputVariable(new int[] { numOutputClasses }, DataType.Float);
                        var loss = CNTKLib.BinaryCrossEntropy(model, labelInput);
                        var classificationError = CNTKLib.ClassificationError(new Variable(model), labelInput);

                        // Configure training
                        var learningRatePerSample = new TrainingParameterScheduleDouble(0.001, BatchSize);

                        var vp = new ParameterVector();
                        foreach (Parameter p in model.Parameters()) vp.Add(p);
                        var parameterLearners = new List<Learner>() { CNTKLib.AdaDeltaLearner(vp, learningRatePerSample, 0.95, 1e-07) };

                        var trainer = Trainer.CreateTrainer(model, loss, classificationError, parameterLearners);
                        var evaluator = CNTKLib.CreateEvaluator(loss);

                        // Training loop
                        for (int epoch = 0; epoch < 10; epoch++)
                        {
                            // Updating the text on the button
                            Dispatcher.Invoke(() =>
                            {
                                Train.Content = $"Обучение... Эпоха {epoch + 1}/10";
                            });
                            for (int i = 0; i < trainBatchCount; i++)
                            {
                                var batchData = LoadBatchData(trainImagePaths, trainLabels, width, height, i * BatchSize, BatchSize).ToList();

                                // Prepare batch
                                var imageTrainBatch = Value.CreateBatch(imageInput.Shape, batchData.SelectMany(x => x.Image).ToList(), device);
                                var labelTrainBatch = Value.CreateBatch(labelInput.Shape, batchData.Select(x => x.Label).ToList(), device);
                                var inputTrainDataMap = new UnorderedMapVariableValuePtr() { { imageInput, imageTrainBatch }, { labelInput, labelTrainBatch } };

                                // Train on batch
                                trainer.TrainMinibatch(inputTrainDataMap, false, device);

                                Console.WriteLine($"Обучающая эпоха {epoch} с пакетом {i + 1}/{trainBatchCount}");
                            }

                            // Evaluation loop
                            double totalTestError = 0;
                            for (int i = 0; i < testCount; i += BatchSize)
                            {
                                var testBatch = LoadBatchData(testImagePaths, testLabels, width, height, i, BatchSize).ToList();

                                var imageTestBatch = Value.CreateBatch(imageInput.Shape, testBatch.SelectMany(x => x.Image).ToList(), device);
                                var labelTestBatch = Value.CreateBatch(labelInput.Shape, testBatch.Select(x => x.Label).ToList(), device);
                                var inputTestDataMap = new UnorderedMapVariableValuePtr() { { imageInput, imageTestBatch }, { labelInput, labelTestBatch } };

                                // Get evaluation result
                                double testError = evaluator.TestMinibatch(inputTestDataMap, device);
                                totalTestError += testError * testBatch.Count;
                                Console.WriteLine($"Ошибка теста для пакета {i / BatchSize + 1}: {testError}");
                            }
                            totalTestError /= testCount;

                            // Save the model after each epoch
                            string modelsDir = System.IO.Path.Combine(baseDir, "Models");
                            if (!Directory.Exists(modelsDir))
                            {
                                Directory.CreateDirectory(modelsDir);
                            }

                            string modelPath = System.IO.Path.Combine(modelsDir, $"model_epoch_{epoch + 1}_testerror_{totalTestError}.dnn");
                            model.Save(modelPath);

                            Console.WriteLine($"Эпоха {epoch + 1} завершена. Модель сохранена в {modelPath}. Ошибка теста: {totalTestError}");
                        }
                        // After successful training, we update the text on the button
                        Dispatcher.Invoke(() =>
                        {
                            Train.Content = "Обучить модель";
                        });

                        // After successful training, we open the folder with models
                        OpenModelsFolder();
                    }, _cancellationTokenSource.Token);
                }
                catch (OperationCanceledException)
                {
                    _cancellationTokenSource = null;

                    // We restore the activity of the interface
                    EnableUI();

                    // We restore the button color to the original
                    Train.Background = _originalButtonColor;

                    _isTraining = false;
                }
                finally
                {
                    _cancellationTokenSource = null;

                    // We restore the activity of the interface
                    EnableUI();

                    // We restore the button color to the original
                    Train.Background = _originalButtonColor;

                    _isTraining = false;
                }
            }
            else
            {
                // We interrupt the training and restore the interface activity
                _cancellationTokenSource?.Cancel();
            }
        }
        private void DisableUI()
        {
            // Disable the buttons and lists on the interface, except for the training button
            OpenSourceDir.IsEnabled = false;
            CloseSourceDir.IsEnabled = false;
            OpenSaveSamplesDir.IsEnabled = false;
            ConvertExistingGrayscale.IsEnabled = false;
            RewriteMarkup.IsEnabled = false;
            RenameSamples.IsEnabled = false;
            SaveSamples.IsEnabled = false;

            // And other buttons and lists that need to be disabled
        }

        private void EnableUI()
        {
            // Enable the buttons and lists on the interface
            OpenSourceDir.IsEnabled = true;
            CloseSourceDir.IsEnabled = true;
            OpenSaveSamplesDir.IsEnabled = true;
            ConvertExistingGrayscale.IsEnabled = true;
            RewriteMarkup.IsEnabled = true;
            RenameSamples.IsEnabled = true;
            SaveSamples.IsEnabled = true;

            // And other buttons and lists that need to be enabled
        }
    }
}