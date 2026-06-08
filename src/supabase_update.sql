-- ============================================================
-- APEX BOT - Migración: soporte para client_order_id
-- Ejecuta en: Supabase Dashboard > SQL Editor
-- ============================================================

-- 1. Añadir columna a trade_state (guarda el ID de la compra activa)
ALTER TABLE trade_state
  ADD COLUMN IF NOT EXISTS client_order_id TEXT DEFAULT NULL;

-- 2. Añadir columna a trade_log (auditoría por orden)
ALTER TABLE trade_log
  ADD COLUMN IF NOT EXISTS client_order_id TEXT DEFAULT NULL;

-- Índice opcional para buscar rápido por ID de orden
CREATE INDEX IF NOT EXISTS idx_trade_log_client_order_id
  ON trade_log (client_order_id);