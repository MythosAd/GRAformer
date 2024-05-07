import matplotlib.pyplot as plt
import pandas as pd
import torch

from gluonts.dataset.common import ListDataset
from gluonts.dataset.util import to_pandas

from pts.model.deepar import DeepAREstimator
from pts import Trainer

url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv"
df = pd.read_csv(url, header=0, index_col=0, parse_dates=True)

df[:100].plot(linewidth=2)
plt.grid(which='both')
plt.show()

training_data = ListDataset(
    [{"start": df.index[0], "target": df.value[:"2015-04-05 00:00:00"]}],
    freq = "5min"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

estimator = DeepAREstimator(freq="5min",
                            prediction_length=12,
                            input_size=19,
                            trainer=Trainer(epochs=10,
                                            device=device))
if __name__ == '__main__':
    predictor = estimator.train(training_data=training_data, num_workers=1)

    test_data = ListDataset(
        [{"start": df.index[0], "target": df.value[:"2015-04-15 00:00:00"]}],
        freq = "5min"
    )

    for test_entry, forecast in zip(test_data, predictor.predict(test_data)):
        to_pandas(test_entry)[-60:].plot(linewidth=2)
        forecast.plot(color='g', prediction_intervals=[50.0, 90.0])
    plt.grid(which='both')