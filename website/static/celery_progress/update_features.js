function updateFeatures(myDict) {
  document.getElementById("loadButton").style.display = "block";
  document.getElementById("resetZoomButtonCorro").style.display = "block";
  document.getElementById("resetZoomButtonPrice").style.display = "block";
  document.getElementById("resetZoomButtonVolume").style.display = "block";
  document.getElementById("resetZoomButtonTrainLoss").style.display = "block";
  document.getElementById("abort").style.display = "none";

  document.getElementById("model_type").textContent = myDict.model_type;
  document.getElementById("train_loss").textContent = myDict.train_loss;
  document.getElementById("test_loss").textContent = myDict.test_loss;
  document.getElementById("features_used").textContent = myDict.features_used;
  document.getElementById("RMSE").textContent = myDict.RMSE;
  document.getElementById("next_predicted_close").textContent =
    myDict.next_day_close;
  document.getElementById("r_2").textContent = myDict.r_squared;
  document.getElementById("r_2_naive").textContent = myDict.r_squared_naive;
  document.getElementById("naive_model_RMSE").textContent = myDict.RMSE_naive;

  if (myDict.beat_naive) {
    document.getElementById("naive_model_beat").textContent = "Yes";
    document.getElementById("naive_model_beat").style.color = "green";
  } else {
    document.getElementById("naive_model_beat").textContent = "No";
    document.getElementById("naive_model_beat").style.color = "red";
  }
  document.getElementById("naive_model_beat").style.fontWeight = "bold";

  const news_inputs = [
    document.getElementById("news_input_1"),
    document.getElementById("news_input_2"),
    document.getElementById("news_input_3"),
    document.getElementById("news_input_4"),
    document.getElementById("news_input_5"),
    document.getElementById("news_input_6"),
    document.getElementById("news_input_7"),
    document.getElementById("news_input_8"),
  ];

  const date_news = [
    document.getElementById("date_news_1"),
    document.getElementById("date_news_2"),
    document.getElementById("date_news_3"),
    document.getElementById("date_news_4"),
    document.getElementById("date_news_5"),
    document.getElementById("date_news_6"),
    document.getElementById("date_news_7"),
    document.getElementById("date_news_8"),
  ];

  const length = Math.min(news_inputs.length, date_news.length);

  for (let i = 0; i < length; i++) {
    news_inputs[i].textContent = myDict.news_headline[i];
    date_news[i].textContent = myDict.news_dates[i];

    if (myDict.news_sentiments[i] == 0) {
      news_inputs[i].style.color = "white";
    } else if (myDict.news_sentiments[i] == 1) {
      news_inputs[i].style.color = "#90EE90";
    } else if (myDict.news_sentiments[i] == -1) {
      news_inputs[i].style.color = "#ff7066";
    }
  }

  const plugin = {
    id: "customCanvasBackgroundColor",
    beforeDraw: (chart, args, options) => {
      const { ctx } = chart;
      ctx.save();
      ctx.globalCompositeOperation = "destination-over";
      ctx.fillStyle = options.color || "#99ffff";
      ctx.fillRect(0, 0, chart.width, chart.height);
      ctx.restore();
    },
  };

  if (window.priceChart) {
    window.priceChart.destroy();
  }
  if (window.volumeChart) {
    window.volumeChart.destroy();
  }
  if (window.corroChart) {
    window.corroChart.destroy();
  }
  if (window.train_loss_chart) {
    window.train_loss_chart.destroy();
  }

  let ctx_price = document.getElementById("price_graph").getContext("2d");
  window.priceChart = new Chart(ctx_price, {
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
          order: 3,
        },
        {
          label: "Naive Model",
          data: myDict.naive_dates.map((x, index) => ({
            x: x,
            y: myDict.naive_close[index],
          })),
          borderColor: "rgba(255,140,0, 1)",
          backgroundColor: "rgba(255,140,0, 0.2)",
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
          grid: {
            color: "#3b3b3b",
          },
          ticks: {
            font: {
              size: 8, // Set this to the desired font size for the x-axis labels
            },
            color: "white",
          },
        },
        y: {
          grid: {
            color: "#3b3b3b",
          },
          ticks: {
            color: "white",
          },
        },
      },
      plugins: {
        customCanvasBackgroundColor: {
          color: "black",
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
        legend: {
          display: true,
          position: "chartArea", // Position inside the chart area
          align: "start", // Align to the start of the chart area
          labels: {
            color: "white",
            usePointStyle: true, // Use point styles
            font: {
              size: 12, // Adjust the font size
            },
            boxWidth: 20, // Adjust the box width
            padding: 10, // Adjust the padding
          },
          layout: {
            padding: {
              top: 5,
            },
          },
        },
      },
    },
    plugins: [plugin],
  });

  const barColors = myDict.close_prices.map((close, index) => {
    return close >= myDict.open_prices[index] ? "blue" : "red";
  });
  let ctx_volume = document.getElementById("volume_graph").getContext("2d");
  window.volumeChart = new Chart(ctx_volume, {
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
      tooltips: {
        callbacks: {
          label: function (tooltipItem, data) {
            // Customize tooltip label to show 5 decimal places
            return (
              data.datasets[tooltipItem.datasetIndex].label +
              ": " +
              tooltipItem.yLabel.toFixed(5)
            );
          },
        },
      },
      plugins: {
        title: {
          display: true,
          text: "Trading Volume by Date",
          font: {
            size: 12,
          },
          color: "white",
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
          grid: {
            color: "#3b3b3b",
          },
          ticks: {
            font: {
              size: 10, // Set this to the desired font size for the x-axis labels
            },
            color: "white",
          },
        },
        x: {
          grid: {
            color: "#3b3b3b",
          },
          ticks: {
            font: {
              size: 8, // Set this to the desired font size for the x-axis labels
            },
            color: "white",
          },
        },
      },
    },
  });

  const barColors_corro = myDict.corrolation_values.map((corro, index) => {
    if (corro > 0.5) {
      return "green";
    } else {
      return "red";
    }
  });

  let ctx_corro = document.getElementById("corrolation_graph").getContext("2d");
  window.corroChart = new Chart(ctx_corro, {
    type: "bar",
    data: {
      labels: myDict.corro_features,
      datasets: [
        {
          data: myDict.corrolation_values,
          backgroundColor: barColors_corro,
          borderColor: barColors_corro,
          borderWidth: 1,
        },
      ],
    },
    options: {
      plugins: {
        title: {
          display: true,
          text: "Previous Day Feature Corrolations with Next Day Closing Price",
          font: {
            size: 12,
          },
          color: "white",
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
          grid: {
            color: "#3b3b3b",
          },
          ticks: {
            color: "white",
          },
        },
        x: {
          grid: {
            color: "#3b3b3b",
          },
          ticks: {
            font: {
              size: 12, // Set this to the desired font size for the x-axis labels
            },
            color: "white",
          },
        },
      },
    },
  });

  let ctx_train_loss = document
    .getElementById("train_loss_graph")
    .getContext("2d");
  window.train_loss_chart = new Chart(ctx_train_loss, {
    type: "line", // or any other type of chart
    data: {
      labels: myDict.train_array.map((_, index) => index),
      datasets: [
        {
          label: "Train Loss",
          data: myDict.train_array,
          borderColor: "rgba(75, 192, 192, 1)",
          backgroundColor: "rgba(75, 192, 192, 0.2)",
          showLine: true,
        },
        {
          label: "Test Loss",
          data: myDict.test_array,
          borderColor: "rgba(255, 99, 132, 1)",
          backgroundColor: "rgba(255, 99, 132, 0.2)",
          showLine: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: {
            color: "#3b3b3b",
          },
          ticks: {
            title: {
              display: true,
              text: "Epoch",
            },
            font: {
              size: 8, // Set this to the desired font size for the x-axis labels
            },
            color: "white",
          },
        },
        y: {
          max:
            Math.max(
              Math.max(...myDict.test_array),
              Math.max(...myDict.train_array)
            ) + 0.3,
          grid: {
            color: "#3b3b3b",
          },
          ticks: {
            color: "white",
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
        legend: {
          display: true,
          position: "chartArea", // Position inside the chart area
          align: "start", // Align to the start of the chart area
          labels: {
            color: "white",
            usePointStyle: true, // Use point styles
            font: {
              size: 10, // Adjust the font size
            },
            boxWidth: 20, // Adjust the box width
            padding: 10, // Adjust the padding
          },
        },
        layout: {
          padding: {
            top: 5,
          },
        },
      },
    },
  });
}
