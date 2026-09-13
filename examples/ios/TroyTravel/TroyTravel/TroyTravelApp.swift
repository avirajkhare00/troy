import MLX
import SwiftUI

@main
struct TroyTravelApp: App {
    init() {
        // Keep MLX's buffer cache small on iOS — memory returned to the cache
        // still counts against the app's jetsam limit.
        MLX.Memory.cacheLimit = 20 * 1024 * 1024
        print("[troytravel] build trace-v7 2026-09-13")
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
