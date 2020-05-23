using Microsoft.ML.Data;
using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ChisoftMLApp1
{
    public  class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public CodeCompileUnit PredictedClusterId;

        [ColumnName("Score")]
        public float[] Distance;
    }
}
