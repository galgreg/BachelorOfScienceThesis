﻿using System;
using System.IO;
using MLAgents.CommunicatorObjects;
using UnityEditor;
using UnityEngine;


namespace MLAgents
{
    /// <summary>
    /// Asset Importer used to parse demonstration files.
    /// </summary>
    [UnityEditor.AssetImporters.ScriptedImporter(1, new[] {"demo"})]
    public class DemonstrationImporter : UnityEditor.AssetImporters.ScriptedImporter
    {
        private const string IconPath = "Assets/ML-Agents/Resources/DemoIcon.png";

        public override void OnImportAsset(UnityEditor.AssetImporters.AssetImportContext ctx)
        {
            var inputType = Path.GetExtension(ctx.assetPath);
            if (inputType == null)
            {
                throw new Exception("Demonstration import error.");
            }

            try
            {
                // Read first two proto objects containing metadata and brain parameters.
                Stream reader = File.OpenRead(ctx.assetPath);

                var metaDataProto = DemonstrationMetaProto.Parser.ParseDelimitedFrom(reader);
                var metaData = new DemonstrationMetaData(metaDataProto);

                reader.Seek(DemonstrationStore.MetaDataBytes + 1, 0);
                var brainParamsProto = BrainParametersProto.Parser.ParseDelimitedFrom(reader);
                var brainParameters = new BrainParameters(brainParamsProto);

                reader.Close();

                var demonstration = ScriptableObject.CreateInstance<Demonstration>();
                demonstration.Initialize(brainParameters, metaData);
                userData = demonstration.ToString();

                Texture2D texture = (Texture2D)
                    AssetDatabase.LoadAssetAtPath(IconPath, typeof(Texture2D));

#if UNITY_2017_3_OR_NEWER
                ctx.AddObjectToAsset(ctx.assetPath, demonstration, texture);
                ctx.SetMainObject(demonstration);
#else
            ctx.SetMainAsset(ctx.assetPath, model);
#endif
            }
            catch
            {
                return;
            }
        }
    }
}
