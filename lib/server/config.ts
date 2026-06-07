export interface ServerConfig {
  pythonPath: string;
  engineDir: string;
  radarCsvPath: string;
  modelDir: string;
}

export function getServerConfig(): ServerConfig {
  return {
    pythonPath: process.env.PYTHON_PATH || "python3",
    engineDir: process.env.ENGINE_DIR || "./lib/engine",
    radarCsvPath:
      process.env.RADAR_CSV_PATH || "/root/quant_bot/hibt_ticks.csv",
    modelDir: process.env.MODEL_DIR || "/root/quant_bot",
  };
}
