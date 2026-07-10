import { AllCommunityModule, ModuleRegistry, themeQuartz, colorSchemeDark } from "ag-grid-community"

ModuleRegistry.registerModules([AllCommunityModule])

const base = themeQuartz.withParams({ spacing: 6, fontSize: 13 })

export const gridThemeLight = base
export const gridThemeDark = base.withPart(colorSchemeDark)
