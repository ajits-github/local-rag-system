import { afterEach, describe, expect, it, vi } from "vitest";
import { getUiModeConfig, resolveUiMode, SAFE_DEFAULT_UI_MODE } from "../../src/config/uiMode";

describe("resolveUiMode", () => {
  it("resolves the literal string 'developer' to developer mode", () => {
    expect(resolveUiMode("developer")).toBe("developer");
  });

  it.each([undefined, "", "production", "Developer", "DEV", "dev", "developer ", "true"])(
    "falls back to the safe default for %j",
    (raw) => {
      expect(resolveUiMode(raw)).toBe(SAFE_DEFAULT_UI_MODE);
    }
  );

  it("documents production as the safe default", () => {
    expect(SAFE_DEFAULT_UI_MODE).toBe("production");
  });
});

describe("getUiModeConfig", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("enables every developer flag when VITE_UI_MODE=developer", () => {
    vi.stubEnv("VITE_UI_MODE", "developer");
    const config = getUiModeConfig();
    expect(config).toEqual({
      uiMode: "developer",
      showDeveloperSettings: true,
      showDebugDetails: true,
      showRuntimeDetails: true,
      showDeveloperModeIndicator: true,
    });
  });

  it("disables every developer flag when VITE_UI_MODE is unset", () => {
    vi.stubEnv("VITE_UI_MODE", undefined);
    const config = getUiModeConfig();
    expect(config).toEqual({
      uiMode: "production",
      showDeveloperSettings: false,
      showDebugDetails: false,
      showRuntimeDetails: false,
      showDeveloperModeIndicator: false,
    });
  });

  it("disables every developer flag for a malformed value", () => {
    vi.stubEnv("VITE_UI_MODE", "developer-mode-please");
    const config = getUiModeConfig();
    expect(config.uiMode).toBe("production");
    expect(config.showDeveloperSettings).toBe(false);
    expect(config.showDebugDetails).toBe(false);
    expect(config.showRuntimeDetails).toBe(false);
    expect(config.showDeveloperModeIndicator).toBe(false);
  });

  it("re-reads the env on every call, so stubbing takes effect without re-importing the module", () => {
    vi.stubEnv("VITE_UI_MODE", "developer");
    expect(getUiModeConfig().uiMode).toBe("developer");

    vi.stubEnv("VITE_UI_MODE", "production");
    expect(getUiModeConfig().uiMode).toBe("production");
  });
});
