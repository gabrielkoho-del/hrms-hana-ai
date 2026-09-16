/**
 * ChartCanvas — renders a Chart.js chart from an hr_chart side-channel payload.
 *
 * The payload shape (produced by agent/output/chart_payload.py):
 *   { type, title, labels, datasets, options }
 *
 * Chart.js is loaded dynamically to avoid blocking the initial bundle.
 */
import React, { useEffect, useRef } from "react";

export default function ChartCanvas({ config }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!config || !canvasRef.current) return;

    let Chart = null;
    let isMounted = true;

    // Dynamically import Chart.js only when we have a chart to render.
    import("chart.js").then(({
      Chart: ChartJS,
      registerables,
      CategoryScale,
      LinearScale,
      PointElement,
      LineElement,
      BarElement,
      ArcElement,
      RadialLinearScale,
      Title,
      Tooltip,
      Legend,
      Filler,
    }) => {
      if (!isMounted) return;
      Chart = ChartJS;
      Chart.register(
        CategoryScale,
        LinearScale,
        PointElement,
        LineElement,
        BarElement,
        ArcElement,
        RadialLinearScale,
        Title,
        Tooltip,
        Legend,
        Filler,
      );

      const ctx = canvasRef.current.getContext("2d");
      if (chartRef.current) chartRef.current.destroy();

      const chartConfig = {
        type: config.type,
        data: {
          labels: config.labels,
          datasets: config.datasets,
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            title: {
              display: !!config.title,
              text: config.title,
              font: { size: 14 },
            },
            legend: {
              display: config.datasets.length > 1 ||
                ["pie", "doughnut", "radar", "polarArea"].includes(config.type),
            },
          },
          ...config.options,
        },
      };

      chartRef.current = new Chart(ctx, chartConfig);
    });

    return () => {
      isMounted = false;
      if (chartRef.current) chartRef.current.destroy();
    };
  }, [config]);

  if (!config) return null;

  return (
    <div style={{
      margin: "12px 0",
      padding: "12px",
      background: "#F7F6F1",
      borderRadius: "12px",
      border: "1px solid #E3E1D6",
      minHeight: "240px",
      position: "relative",
    }}>
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%" }} />
    </div>
  );
}
