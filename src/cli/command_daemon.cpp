#include "cli/command_daemon.h"
#include "cli/systemd_client.h"
#include "common/i18n.h"
#include "common/string_utils.h"

int RunDaemonStart(Formatter& fmt, const CliContext& ctx) {
    (void)ctx;
    int r = vinput::cli::SystemctlStart();
    if (r == 0) {
        fmt.PrintSuccess(_("Daemon started."));
        return 0;
    }
    fmt.PrintError(vinput::str::FmtStr(_("systemctl start failed (exit code: %d)"), r));
    return 1;
}

int RunDaemonStop(Formatter& fmt, const CliContext& ctx) {
    (void)ctx;
    int r = vinput::cli::SystemctlStop();
    if (r == 0) {
        fmt.PrintSuccess(_("Daemon stopped."));
        return 0;
    }
    fmt.PrintError(vinput::str::FmtStr(_("systemctl stop failed (exit code: %d)"), r));
    return 1;
}

int RunDaemonRestart(Formatter& fmt, const CliContext& ctx) {
    (void)ctx;
    int r = vinput::cli::SystemctlRestart();
    if (r == 0) {
        fmt.PrintSuccess(_("Daemon restarted."));
        return 0;
    }
    fmt.PrintError(vinput::str::FmtStr(_("systemctl restart failed (exit code: %d)"), r));
    return 1;
}

int RunDaemonLogs(bool follow, int lines, Formatter& fmt, const CliContext& ctx) {
    (void)fmt;
    (void)ctx;
    return vinput::cli::JournalctlLogs(follow, lines);
}
