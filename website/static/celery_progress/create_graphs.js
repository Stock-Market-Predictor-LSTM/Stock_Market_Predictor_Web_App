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

let ctx_price = document.getElementById("price_graph").getContext("2d");
window.priceChart = new Chart(ctx_price, {
  type: "line", // or any other type of chart
  data: {
    labels: [0, 1],
    datasets: [
      {
        label: "Waiting For Data",
        data: [0, 0],
        borderColor: "rgba(0, 151, 255, 1)",
        backgroundColor: "rgba(0, 151, 255, 0.2)",
        showLine: true,
        order: 1,
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
            size: 10, // Adjust the font size
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

let ctx_volume = document.getElementById("volume_graph").getContext("2d");
window.volumeChart = new Chart(ctx_volume, {
  type: "bar",
  data: {
    labels: ["Waiting For Data"],
    datasets: [
      {
        data: [0],
        backgroundColor: "red",
        borderColor: "red",
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
            size: 12, // Set this to the desired font size for the x-axis labels
          },
          color: "white",
        },
      },
    },
  },
});

let ctx_corro = document.getElementById("corrolation_graph").getContext("2d");
window.corroChart = new Chart(ctx_corro, {
  type: "bar",
  data: {
    labels: ["Waiting For Data"],
    datasets: [
      {
        data: [0],
        backgroundColor: "red",
        borderColor: "red",
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
    labels: [0, 1],
    datasets: [
      {
        label: "Waiting For Data",
        data: [0, 0],
        borderColor: "rgba(75, 192, 192, 1)",
        backgroundColor: "rgba(75, 192, 192, 0.2)",
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
            size: 12, // Adjust the font size
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
