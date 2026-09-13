import MLX
import SwiftUI

@main
struct TroyChatApp: App {
    init() {
        // Keep MLX's buffer cache small on iOS — memory returned to the cache
        // still counts against the app's jetsam limit.
        MLX.Memory.cacheLimit = 20 * 1024 * 1024
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
