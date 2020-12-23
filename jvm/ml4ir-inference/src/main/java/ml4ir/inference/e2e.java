package ml4ir.inference;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.protobuf.GeneratedMessageV3;
import ml4ir.inference.tensorflow.ModelExecutorConfig;
import ml4ir.inference.tensorflow.TFRecordExecutor;
import ml4ir.inference.tensorflow.data.ModelFeaturesConfig;
import ml4ir.inference.tensorflow.data.SequenceExampleBuilder;
import ml4ir.inference.tensorflow.data.StringMapSequenceExampleBuilder;
import org.tensorflow.example.FeatureList;
import org.tensorflow.example.SequenceExample;

import java.util.List;
import java.util.Map;

public class e2e {

    private static final String FEATURE_CONFIG = "python/feature_config_test_e2e.yaml";
    private static final String DATASET = "python/TEST_DIR/validation/validation.csv";
    private static final String MODEL = "python/models/e2e_run/final/tfrecord";

    public static void main(String[] args) {
        final String PROJECT_DIR = args[0];

        Map<String, String> contextMap = ImmutableMap.of("query_id", "q00");
        List<Map<String, String>> documents = ImmutableList.of(
                ImmutableMap.of(
                        "feature1", "474",
                        "feature2", "0.39254769476227647",
                        "feature3", "wibdifgefxkzluocsrwerwixmzutpwgfw"
                ),
                ImmutableMap.of(
                        "feature1", "339",
                        "feature2", "0.9993560517954271",
                        "feature3", "qzwjdmiwtiqlxwywajtfcyouxssuoccdyskalhyhrdhztfyhh"
                )
        );
        ModelFeaturesConfig modelFeatures = ModelFeaturesConfig.load(PROJECT_DIR + FEATURE_CONFIG);

        SequenceExampleBuilder<Map<String, String>, Map<String, String>> sequenceExampleBuilder =
                StringMapSequenceExampleBuilder.withFeatureProcessors(
                        modelFeatures,
                        ImmutableMap.of(),  // No processing for int, float or string
                        ImmutableMap.of(),
                        ImmutableMap.of());

        SequenceExample sequenceExample = sequenceExampleBuilder.build(contextMap, documents);


        ModelExecutorConfig modelConfig = new ModelExecutorConfig("serving_tfrecord_protos",
                "StatefulPartitionedCall");
        TFRecordExecutor bundleExecutor = new TFRecordExecutor(PROJECT_DIR + MODEL, modelConfig);

        Map<String, FeatureList> featureListMap = sequenceExample.getFeatureLists().getFeatureListMap();

        List<Long> feature1 =
                featureListMap.get("feature1").getFeatureList().get(0).getInt64List().getValueList();

        List<Float> feature2 =
                featureListMap.get("feature2").getFeatureList().get(0).getFloatList().getValueList();

        float[] predictions = bundleExecutor.apply(sequenceExample);

        System.out.println(predictions);

    }

}
