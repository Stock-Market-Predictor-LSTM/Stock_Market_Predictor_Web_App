class CeleryProgressBar {
  constructor(progressUrl, options) {
    this.progressUrl = progressUrl;
    options = options || {};
    let progressBarId = options.progressBarId || "progress-bar";
    let progressBarMessage =
      options.progressBarMessageId || "progress-bar-message";
    this.progressBarElement =
      options.progressBarElement || document.getElementById(progressBarId);
    this.progressBarMessageElement =
      options.progressBarMessageElement ||
      document.getElementById(progressBarMessage);
    this.onProgress = options.onProgress || this.onProgressDefault;
    this.onSuccess = options.onSuccess || this.onSuccessDefault;
    this.onError = options.onError || this.onErrorDefault;
    this.onTaskError = options.onTaskError || this.onTaskErrorDefault;
    this.onDataError = options.onDataError || this.onError;
    this.onRetry = options.onRetry || this.onRetryDefault;
    this.onIgnored = options.onIgnored || this.onIgnoredDefault;
    let resultElementId = options.resultElementId || "celery-result";
    this.resultElement =
      options.resultElement || document.getElementById(resultElementId);
    this.onResult = options.onResult || this.onResultDefault;
    // HTTP options
    this.onNetworkError = options.onNetworkError || this.onError;
    this.onHttpError = options.onHttpError || this.onError;
    this.pollInterval = options.pollInterval || 400;
    this.maxNetworkRetryAttempts = options.maxNetworkRetryAttempts | 5;
    // Other options
    this.barColors = Object.assign(
      {},
      this.constructor.getBarColorsDefault(),
      options.barColors
    );

    let defaultMessages = {
      waiting: "Waiting for data to be loaded...",
      started: "Task started...",
    };
    this.messages = Object.assign({}, defaultMessages, options.defaultMessages);
  }

  onSuccessDefault(progressBarElement, progressBarMessageElement, result) {
    result = this.getMessageDetails(result);
    if (progressBarElement) {
      progressBarElement.style.backgroundColor = this.barColors.success;
    }
    if (progressBarMessageElement) {
      progressBarMessageElement.textContent = "Sucessfuly trained and loaded. ";
      document.getElementById("loadButton").style.display = "block";
      document.getElementById("abort").style.display = "none";

      var myDict = result;
      let ctx_price = document.getElementById("price_graph").getContext("2d");

      var priceChart = new Chart(ctx_price, {
        type: "line", // or any other type of chart
        data: {
          labels: myDict.dates,
          datasets: [
            {
              label: myDict.ticker + " Close Prices",
              data: myDict.dates.map((x, index) => ({
                x: x,
                y: myDict.close_prices[index],
              })),
              borderColor: "rgba(0, 151, 255, 1)",
              backgroundColor: "rgba(0, 151, 255, 0.2)",
              showLine: true,
              order: 2,
            },
            {
              label: "Train Data Prediction",
              data: myDict.x_axis_close_train.map((x, index) => ({
                x: x,
                y: myDict.y_axis_close_train[index],
              })),
              borderColor: "rgba(255, 88, 0, 1)",
              backgroundColor: "rgba(255, 88, 0, 0.2)",
              showLine: true,
              order: 1,
            },
            {
              label: "Test Data Prediction",
              data: myDict.x_axis_close_test.map((x, index) => ({
                x: x,
                y: myDict.y_axis_close_test[index],
              })),
              borderColor: "rgba(0, 255, 19, 1)",
              backgroundColor: "rgba(0, 255, 19, 0.2)",
              showLine: true,
              order: 0,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              ticks: {
                font: {
                  size: 8, // Set this to the desired font size for the x-axis labels
                },
              },
            },
          },
          plugins: {
            zoom: {
              zoom: {
                wheel: {
                  enabled: true, // Enable zooming with the mouse wheel
                },
                pinch: {
                  enabled: true, // Enable zooming with pinch gestures
                },
                mode: "xy", // Allow zooming in both directions
              },
              pan: {
                enabled: true, // Enable panning
                mode: "xy", // Allow panning in both directions
              },
            },
          },
        },
      });

      const barColors = myDict.close_prices.map((close, index) => {
        return close >= myDict.open_prices[index] ? "blue" : "red";
      });
      let ctx_volume = document.getElementById("volume_graph").getContext("2d");
      var volumeChart = new Chart(ctx_volume, {
        type: "bar",
        data: {
          labels: myDict.dates,
          datasets: [
            {
              data: myDict.volume,
              backgroundColor: barColors,
              borderColor: barColors,
              borderWidth: 1,
            },
          ],
        },
        options: {
          plugins: {
            title: {
              display: true,
              text: "Trading Volume by Date",
              font: {
                size: 12,
              },
            },
            legend: {
              display: false,
            },
            zoom: {
              zoom: {
                wheel: {
                  enabled: true, // Enable zooming with the mouse wheel
                },
                pinch: {
                  enabled: true, // Enable zooming with pinch gestures
                },
                mode: "xy", // Allow zooming in both directions
              },
              pan: {
                enabled: true, // Enable panning
                mode: "xy", // Allow panning in both directions
              },
            },
          },
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
            },
            x: {
              ticks: {
                font: {
                  size: 8, // Set this to the desired font size for the x-axis labels
                },
              },
            },
          },
        },
      });

      let ctx_macd = document.getElementById("macd_graph").getContext("2d");

      var macdChart = new Chart(ctx_macd, {
        type: "line", // or any other type of chart
        data: {
          labels: myDict.dates,
          datasets: [
            {
              label: "Macd",
              data: myDict.macd,
              borderColor: "rgba(75, 192, 192, 1)",
              backgroundColor: "rgba(75, 192, 192, 0.2)",
            },
            {
              label: "Signal Line",
              data: myDict.signal_line,
              borderColor: "rgba(192, 75, 75, 1)",
              backgroundColor: "rgba(192, 75, 75, 0.2)",
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              ticks: {
                font: {
                  size: 8, // Set this to the desired font size for the x-axis labels
                },
              },
            },
          },
          plugins: {
            zoom: {
              zoom: {
                wheel: {
                  enabled: true, // Enable zooming with the mouse wheel
                },
                pinch: {
                  enabled: true, // Enable zooming with pinch gestures
                },
                mode: "xy", // Allow zooming in both directions
              },
              pan: {
                enabled: true, // Enable panning
                mode: "xy", // Allow panning in both directions
              },
            },
          },
        },
      });
    }
  }

  onResultDefault(resultElement, result) {
    if (resultElement) {
      resultElement.textContent = result;
    }
  }

  /**
   * Default handler for all errors.
   * @param data - A Response object for HTTP errors, undefined for other errors
   */
  onErrorDefault(
    progressBarElement,
    progressBarMessageElement,
    excMessage,
    data
  ) {
    progressBarElement.style.backgroundColor = this.barColors.error;
    excMessage = excMessage || "";
    progressBarMessageElement.textContent =
      "Uh-Oh, something went wrong! " + excMessage;
    document.getElementById("loadButton").style.display = "block";
    document.getElementById("abort").style.display = "none";
  }

  onTaskErrorDefault(
    progressBarElement,
    progressBarMessageElement,
    excMessage
  ) {
    let message = this.getMessageDetails(excMessage);
    this.onError(progressBarElement, progressBarMessageElement, message);
  }

  onRetryDefault(
    progressBarElement,
    progressBarMessageElement,
    excMessage,
    retrySeconds
  ) {
    let message = "Retrying after " + retrySeconds + "s: " + excMessage;
    progressBarElement.style.backgroundColor = this.barColors.error;
    progressBarMessageElement.textContent = message;
  }

  onIgnoredDefault(progressBarElement, progressBarMessageElement, result) {
    progressBarElement.style.backgroundColor = this.barColors.ignored;
    progressBarMessageElement.textContent = result || "Task result ignored!";
    document.getElementById("loadButton").style.display = "block";
    document.getElementById("abort").style.display = "none";
  }

  onProgressDefault(progressBarElement, progressBarMessageElement, progress) {
    progressBarElement.style.backgroundColor = this.barColors.progress;
    progressBarElement.style.width = progress.percent + "%";
    var description = progress.description || "";
    if (progress.current == 0) {
      if (progress.pending === true) {
        progressBarMessageElement.textContent = this.messages.waiting;
      } else {
        progressBarMessageElement.textContent = this.messages.started;
      }
    } else {
      progressBarMessageElement.textContent =
        ((progress.current / progress.total) * 100).toFixed(2) +
        " % training model" +
        description;
    }
  }

  getMessageDetails(result) {
    if (this.resultElement) {
      return "";
    } else {
      return result || "";
    }
  }

  /**
   * Process update message data.
   * @return true if the task is complete, false if it's not, undefined if `data` is invalid
   */
  onData(data) {
    let done = false;
    if (data.progress) {
      this.onProgress(
        this.progressBarElement,
        this.progressBarMessageElement,
        data.progress
      );
    }
    if (data.complete === true) {
      done = true;
      if (data.success === true) {
        this.onSuccess(
          this.progressBarElement,
          this.progressBarMessageElement,
          data.result
        );
      } else if (data.success === false) {
        if (data.state === "RETRY") {
          this.onRetry(
            this.progressBarElement,
            this.progressBarMessageElement,
            data.result.message,
            data.result.next_retry_seconds
          );
          done = false;
          delete data.result;
        } else {
          this.onTaskError(
            this.progressBarElement,
            this.progressBarMessageElement,
            data.result
          );
        }
      } else {
        if (data.state === "IGNORED") {
          this.onIgnored(
            this.progressBarElement,
            this.progressBarMessageElement,
            data.result
          );
          delete data.result;
        } else {
          done = undefined;
          this.onDataError(
            this.progressBarElement,
            this.progressBarMessageElement,
            "Data Error"
          );
        }
      }
      if (data.hasOwnProperty("result")) {
        this.onResult(this.resultElement, data.result);
      }
    } else if (data.complete === undefined) {
      done = undefined;
      this.onDataError(
        this.progressBarElement,
        this.progressBarMessageElement,
        "Data Error"
      );
    }
    return done;
  }

  async connect() {
    let response;
    let success = false;
    let error = null;
    let attempts = 0;
    while (!success && attempts < this.maxNetworkRetryAttempts) {
      try {
        response = await fetch(this.progressUrl);
        success = true;
      } catch (networkError) {
        document.getElementById("loadButton").style.display = "block";
        document.getElementById("abort").style.display = "none";
        error = networkError;
        this.onNetworkError(
          this.progressBarElement,
          this.progressBarMessageElement,
          "Network Error"
        );
        attempts++;
        await new Promise((r) => setTimeout(r, 1000));
      }
    }

    if (!success) {
      throw error;
    }

    if (response.status === 200) {
      let data;
      try {
        data = await response.json();
      } catch (parsingError) {
        this.onDataError(
          this.progressBarElement,
          this.progressBarMessageElement,
          "Parsing Error"
        );
        throw parsingError;
      }

      const complete = this.onData(data);

      if (complete === false) {
        setTimeout(this.connect.bind(this), this.pollInterval);
      }
    } else {
      this.onHttpError(
        this.progressBarElement,
        this.progressBarMessageElement,
        "HTTP Code " + response.status,
        response
      );
    }
  }

  static getBarColorsDefault() {
    return {
      success: "#76ce60",
      error: "#dc4f63",
      progress: "#68a9ef",
      ignored: "#7a7a7a",
    };
  }

  static initProgressBar(progressUrl, options) {
    const bar = new this(progressUrl, options);
    bar.connect();
  }
}
