var priceDataElement = document.getElementById('price_data');
    if (priceDataElement.textContent == 'null') {
        var myDict = {
            ticker: 'No Ticker',
            close_prices: [0,0],
            dates: ['0','1'],
            volume:[0,0],
            open_prices: [0,0],
            macd: [0,0],
            signal_line: [0,0],
        };
    } else {
        var myDict = JSON.parse(priceDataElement.textContent);
    }

    let ctx_price = document.getElementById("price_graph").getContext("2d");
        
    var priceChart = new Chart(ctx_price, {
                    type: 'line', // or any other type of chart
                    data: {
                        labels: myDict.dates,
                        datasets: [{
                            label: myDict.ticker,
                            data: myDict.close_prices,
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                                x: {
                                    ticks: {
                                        font: {
                                            size: 8 // Set this to the desired font size for the x-axis labels
                                        }
                                    }
                                }
                            },
                        plugins:{
                            zoom: {
                                    zoom: {
                                        wheel: {
                                            enabled: true, // Enable zooming with the mouse wheel
                                        },
                                        pinch: {
                                            enabled: true // Enable zooming with pinch gestures
                                        },
                                        mode: 'xy', // Allow zooming in both directions
                                    },
                                    pan: {
                                        enabled: true, // Enable panning
                                        mode: 'xy', // Allow panning in both directions
                                    }
                                },
                            }
                        }
                    });

    const barColors = myDict.close_prices.map((close, index) => {
                        return close >= myDict.open_prices[index] ? 'blue' : 'red';
                    });
    let ctx_volume = document.getElementById("volume_graph").getContext("2d");
    var volumeChart = new Chart(ctx_volume, {
                        type: 'bar',
                        data: {
                            labels: myDict.dates,
                            datasets: [{
                                data: myDict.volume,
                                backgroundColor: barColors,
                                borderColor: barColors,
                                borderWidth: 1
                            }]
                        },
                        options: {
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Trading Volume by Date',
                                    font: {
                                        size: 12
                                    }
                                },
                                legend: {
                                    display: false
                                },
                                zoom: {
                                    zoom: {
                                        wheel: {
                                            enabled: true, // Enable zooming with the mouse wheel
                                        },
                                        pinch: {
                                            enabled: true // Enable zooming with pinch gestures
                                        },
                                        mode: 'xy', // Allow zooming in both directions
                                    },
                                    pan: {
                                        enabled: true, // Enable panning
                                        mode: 'xy', // Allow panning in both directions
                                    }
                                },
                            },
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    beginAtZero: true
                                },
                                x: {
                                    ticks: {
                                        font: {
                                            size: 8 // Set this to the desired font size for the x-axis labels
                                        }
                                    }
                                }
                            },
                        }
                    });

    let ctx_macd = document.getElementById("macd_graph").getContext("2d");
    
    var macdChart = new Chart(ctx_macd, {
                    type: 'line', // or any other type of chart
                    data: {
                        labels: myDict.dates,
                        datasets: [{
                            label: 'Macd',
                            data: myDict.macd,
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        },
                        {
                            label: 'Signal Line',
                            data: myDict.signal_line,
                            borderColor: 'rgba(192, 75, 75, 1)',
                            backgroundColor: 'rgba(192, 75, 75, 0.2)',
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                                x: {
                                    ticks: {
                                        font: {
                                            size: 8 // Set this to the desired font size for the x-axis labels
                                        }
                                    }
                                }
                            },
                        plugins:{
                            zoom: {
                                    zoom: {
                                        wheel: {
                                            enabled: true, // Enable zooming with the mouse wheel
                                        },
                                        pinch: {
                                            enabled: true // Enable zooming with pinch gestures
                                        },
                                        mode: 'xy', // Allow zooming in both directions
                                    },
                                    pan: {
                                        enabled: true, // Enable panning
                                        mode: 'xy', // Allow panning in both directions
                                    }
                                },
                        }
                    }
                    });