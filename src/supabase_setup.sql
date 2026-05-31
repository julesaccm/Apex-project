-- ============================================================
-- APEX BOT - Setup de Supabase
-- Ejecuta este script en: Supabase Dashboard > SQL Editor
-- ============================================================

-- 1. Estado actual del bot (siempre 1 sola fila, id=1)
CREATE TABLE IF NOT EXISTS trade_state (
    id                   INTEGER PRIMARY KEY DEFAULT 1,
    posicion_abierta     BOOLEAN     DEFAULT FALSE,
    precio_compra        NUMERIC     DEFAULT 0,
    precio_max_alcanzado NUMERIC     DEFAULT 0,
    nivel_stop_loss      NUMERIC     DEFAULT 0,
    cantidad_btc         NUMERIC     DEFAULT 0,
    trailing_activation  NUMERIC     DEFAULT 0.03,
    trailing_distancia   NUMERIC     DEFAULT 0.015,
    updated_at           TIMESTAMPTZ DEFAULT NOW()
);

-- Fila inicial (el bot siempre opera sobre id=1)
INSERT INTO trade_state (id)
VALUES (1)
ON CONFLICT (id) DO NOTHING;

-- 2. Historial de evaluaciones y operaciones (todas las decisiones del bot)
CREATE TABLE IF NOT EXISTS trade_log (
    id              SERIAL PRIMARY KEY,
    tipo            TEXT        NOT NULL,   -- 'COMPRA' | 'VENTA' | 'ESPERA' | 'SENAL_COMPRA' | 'SENAL_VENTA'
    precio          NUMERIC     DEFAULT 0,
    cantidad        NUMERIC     DEFAULT 0,
    ganancia_pct    NUMERIC,               -- Solo para tipo='VENTA'
    senal_prob      JSONB,                  -- {"prob_minimo": 0.72, "prob_maximo": 0.10, ...}
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    -- Foreign key solo poblado para operaciones reales (COMPRA/VENTA)
    -- Evaluaciones (ESPERA/SENAL_*) tienen id_trade_state = NULL
    id_trade_state  INTEGER REFERENCES trade_state(id) ON DELETE CASCADE,
    CONSTRAINT fk_trade_state_only_on_trades 
        CHECK (
            (tipo IN ('COMPRA', 'VENTA') AND id_trade_state IS NOT NULL) OR
            (tipo NOT IN ('COMPRA', 'VENTA') AND id_trade_state IS NULL)
        )
);
