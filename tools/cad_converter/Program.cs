using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using ACadSharp.IO;

internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length != 3)
        {
            Console.Error.WriteLine("Usage: ElectricalBom.CadConverter input.dwg output.dxf report.json");
            return 2;
        }
        try
        {
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
            var notices = new List<object>();
            var counts = new Dictionary<string, int>();
            var doc = DwgReader.Read(args[0], (sender, e) =>
            {
                var type = e.NotificationType.ToString();
                counts[type] = counts.GetValueOrDefault(type) + 1;
                if (notices.Count < 1000)
                    notices.Add(new { type, message = e.Message });
            });
            if (File.Exists(args[1]))
                throw new IOException("Output already exists; use a fresh conversion directory.");
            using (var writer = new DxfWriter(args[1], doc, false))
                writer.Write();
            File.WriteAllText(args[2], JsonSerializer.Serialize(new
            {
                schema_version = 1,
                provider = "ACadSharp",
                version = "3.8.35",
                source_entity_count = doc.Entities.Count(),
                source_entity_types = doc.Entities.GroupBy(e => e.GetType().Name)
                    .ToDictionary(g => g.Key, g => g.Count()),
                notification_counts = counts,
                notifications = notices,
            }), new UTF8Encoding(false));
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.GetType().Name + ": " + ex.Message);
            return 1;
        }
    }
}
